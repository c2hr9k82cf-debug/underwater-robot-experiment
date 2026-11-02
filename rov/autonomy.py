"""自主运动状态机（纯逻辑，无 IO）。

状态与转换：
    SEARCH    找球巡逻        ──检出──▶ ALIGN
    ALIGN     把球心转到画面中央      ──对中──▶ APPROACH
    APPROACH  前进接近 + 持续对中修正  ──够大──▶ TOUCH
                                 └──丢失>0.8s──▶ SEARCH
    TOUCH     低速轻推 1.5s          ──▶ BACKOFF
    BACKOFF   后退 1.5s              ──▶ DONE
    DONE      停桨，任务结束

寻找策略（长方形水池，球从一侧扔到对侧、贴墙）：
    advance   直行前进找球（航向保持 + 贴侧墙向池心修正）
    backoff   撞到对面墙还没见到球 → 先退离墙
    slide     沿墙横移巡逻找球，单侧走完掉头走另一侧（来回）
    子阶段是 SEARCH 内部记忆，检出即转入 ALIGN；目标丢失回 SEARCH 从 advance 重新开始。

姿态传感器（yaw）用法：记住出发航向 → 直行保持航向防跑偏；
横移前转到垂直方向（±90°）、横移中保持。无姿态数据时退化为定时转向。

决策函数是纯函数（输入检测结果+墙壁+姿态+时间 → 输出推力+新状态），
全部可离线单测。本模块不含任何通信/显示代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from rov.config import WALL_FRONT_LIMIT
from rov.detector import DetectionResult, WallSight

# --- 可调参数 ---
SEARCH_SURGE: float = 0.5         # 搜索直行前进速度
AVOID_YAW: float = 0.45           # 避障转向角速度
ALIGN_KP: float = 0.8             # 对中 P 增益（误差→转向角速度）
ALIGN_DONE_ERR: float = 0.12      # |归一化横偏差| 小于该值视为对中
APPROACH_SURGE: float = 0.5       # 接近前进速度
APPROACH_YAW_CAP: float = 0.35    # 接近时对中修正上限
TOUCH_AREA_RATIO: float = 0.12    # 目标面积占画面比例达到该值→轻触
TOUCH_SURGE: float = 0.35         # 轻推速度
TOUCH_S: float = 1.5              # 轻推时长
BACKOFF_SURGE: float = -0.4       # 后退速度
BACKOFF_S: float = 1.5            # 后退时长
LOST_TIMEOUT_S: float = 0.8       # 丢失超过该时长回搜索
LOW_CONF_AREA_RATIO: float = 0.005  # 面积占比低于此视为低置信（只对中不前进）

# --- 沿墙搜索（撞墙后退 → 横移巡逻）---
SLIDE_BACKOFF_S: float = 1.5      # 撞墙后退时长
SLIDE_SURGE: float = 0.35         # 沿墙横移速度
SLIDE_MAX_S: float = 14.0         # 单侧横移时长，超过掉头
TURN_YAW: float = 0.5             # 横移前转向速度（无姿态传感器 fallback）
TURN_KP: float = 0.01             # 转向航向 P 增益（度误差→推力）
TURN_DONE_ERR: float = 12.0       # 转向到位角度容差（度）
HEADING_KP: float = 0.015         # 航向保持增益（度误差→推力）
HEADING_CAP: float = 0.3          # 航向修正上限
SIDE_WALL_LIMIT: float = 0.45     # 侧墙蓝色占比超过此值视为贴墙（跑偏）
SIDE_FIX_YAW: float = 0.2         # 贴侧墙时向池心修正的转向


class AutonomyState(Enum):
    SEARCH = "SEARCH"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    TOUCH = "TOUCH"
    BACKOFF = "BACKOFF"
    DONE = "DONE"


@dataclass
class AutonomyOutput:
    """一次决策的输出。"""

    surge: float
    yaw: float
    heave: float
    state: AutonomyState
    note: str = ""


@dataclass
class AutonomyContext:
    """状态机内部记忆（跨帧保持）。"""

    state: AutonomyState = AutonomyState.SEARCH
    state_since: float = 0.0            # 进入当前状态的时刻
    last_seen_at: float = -1.0          # 最后一次检出时刻
    last_seen_cx: Optional[float] = None  # 最后检出的归一化横坐标
    done: bool = False
    # --- SEARCH 沿墙搜索子阶段 ---
    search_phase: str = "advance"       # advance / backoff / slide
    phase_since: float = 0.0            # 进入当前子阶段的时刻
    heading_ref: Optional[float] = None  # 出发航向（度）
    slide_target: Optional[float] = None  # 横移目标航向（度）
    slide_dir: float = 1.0              # 横移方向 +1 / -1


# ------------------------------------------------------------------ #
# 小工具
# ------------------------------------------------------------------ #

def _norm_cx(result: DetectionResult, frame_w: int) -> float:
    """归一化横偏差 ∈ [-1, 1]，正=球在画面右侧。"""
    assert result.center is not None
    return (result.center[0] - frame_w / 2.0) / (frame_w / 2.0)


def _angle_diff(target: float, current: float) -> float:
    """目标与当前航向的角度差 ∈ [-180, 180]。"""
    return (target - current + 180.0) % 360.0 - 180.0


def _heading_hold(
    target: Optional[float], current: Optional[float], cap: float = HEADING_CAP
) -> float:
    """航向保持：输出把 current 修正到 target 的转向推力。无姿态数据返回 0。"""
    if target is None or current is None:
        return 0.0
    return max(-cap, min(cap, HEADING_KP * _angle_diff(target, current)))


def _reset_search(ctx: AutonomyContext) -> None:
    """回到直行搜索初始状态（保留 slide_dir 历史）。"""
    ctx.search_phase = "advance"
    ctx.phase_since = 0.0
    ctx.heading_ref = None
    ctx.slide_target = None


def step(
    ctx: AutonomyContext,
    result: DetectionResult,
    frame_size: Tuple[int, int],
    now: float,
    wall: Optional[WallSight] = None,
    yaw_deg: Optional[float] = None,
) -> AutonomyOutput:
    """状态机单步决策（ctx 是唯一可变记忆）。

    每帧先做一次状态切换（最多一次），然后输出**新状态**的动作，
    切换帧不停顿。参数：
    - ctx: 跨帧记忆（调用方持有并复用）
    - result: 本帧检测结果
    - frame_size: (宽, 高)
    - now: 单调时钟秒（测试可注入假时间）
    - wall: 蓝色墙壁视野（可选，None=不做避障）
    - yaw_deg: 姿态传感器航向角（度，可选，None=无姿态数据）
    """
    if ctx.state is AutonomyState.DONE:
        return AutonomyOutput(0.0, 0.0, 0.0, AutonomyState.DONE, "任务已完成")

    w, h = frame_size
    frame_area = float(w * h)
    detected = result.is_detected

    # 更新检出记忆
    if detected:
        ctx.last_seen_at = now
        ctx.last_seen_cx = _norm_cx(result, w)

    lost_for = now - ctx.last_seen_at if ctx.last_seen_at >= 0 else float("inf")
    err = ctx.last_seen_cx or 0.0
    area_ratio = (result.area or 0.0) / frame_area

    # ---- 状态切换（每帧最多一次） ----
    note = ""
    if ctx.state is AutonomyState.SEARCH and detected:
        ctx.state = AutonomyState.ALIGN
        ctx.state_since = now
        _reset_search(ctx)
        note = "发现目标，开始对准"
    elif ctx.state in (AutonomyState.ALIGN, AutonomyState.APPROACH) and not detected \
            and lost_for > LOST_TIMEOUT_S:
        ctx.state = AutonomyState.SEARCH
        ctx.state_since = now
        _reset_search(ctx)
        note = "目标丢失，恢复搜索"
    elif ctx.state is AutonomyState.ALIGN and detected and abs(err) < ALIGN_DONE_ERR:
        ctx.state = AutonomyState.APPROACH
        ctx.state_since = now
        note = "已对中，开始接近"
    elif ctx.state is AutonomyState.APPROACH and detected \
            and area_ratio >= TOUCH_AREA_RATIO:
        ctx.state = AutonomyState.TOUCH
        ctx.state_since = now
        note = "够近了，轻触"
    elif ctx.state is AutonomyState.TOUCH and now - ctx.state_since >= TOUCH_S:
        ctx.state = AutonomyState.BACKOFF
        ctx.state_since = now
        note = "轻触完成，后退"
    elif ctx.state is AutonomyState.BACKOFF and now - ctx.state_since >= BACKOFF_S:
        ctx.state = AutonomyState.DONE
        ctx.state_since = now
        ctx.done = True
        note = "任务完成"

    # ---- 避障（只有前方被墙占满且没看到目标才转；目标可见时直接冲目标）----
    # SEARCH 的墙处理由下面的沿墙搜索子阶段负责；ALIGN 不前进；BACKOFF 在后退。
    if (
        wall is not None
        and not detected
        and wall.front >= WALL_FRONT_LIMIT
        and ctx.state in (AutonomyState.APPROACH, AutonomyState.TOUCH)
    ):
        direction = -1.0 if wall.left < wall.right else 1.0
        return AutonomyOutput(
            0.0, direction * AVOID_YAW, 0.0, ctx.state,
            note or "前方被墙挡住，转向避让",
        )

    # ---- 当前状态动作 ----
    if ctx.state is AutonomyState.SEARCH:
        return _search_action(ctx, wall, yaw_deg, now, note)

    if ctx.state is AutonomyState.ALIGN:
        if not detected:
            return AutonomyOutput(0.0, 0.0, 0.0, ctx.state, note or "短暂丢失，保持")
        yaw = max(-0.5, min(0.5, ALIGN_KP * err))
        return AutonomyOutput(0.0, yaw, 0.0, ctx.state, note)

    if ctx.state is AutonomyState.APPROACH:
        if not detected:
            return AutonomyOutput(0.0, 0.0, 0.0, ctx.state, note or "短暂丢失，保持")
        yaw = max(-APPROACH_YAW_CAP, min(APPROACH_YAW_CAP, ALIGN_KP * err))
        if area_ratio < LOW_CONF_AREA_RATIO:
            # 低置信：只对中不前进，防浑水误吸附
            return AutonomyOutput(0.0, yaw, 0.0, ctx.state, note or "低置信，仅对中")
        return AutonomyOutput(APPROACH_SURGE, yaw, 0.0, ctx.state, note)

    if ctx.state is AutonomyState.TOUCH:
        return AutonomyOutput(TOUCH_SURGE, 0.0, 0.0, ctx.state, note)

    if ctx.state is AutonomyState.BACKOFF:
        return AutonomyOutput(BACKOFF_SURGE, 0.0, 0.0, ctx.state, note)

    # DONE 兜底
    return AutonomyOutput(0.0, 0.0, 0.0, AutonomyState.DONE, note)


def _search_action(
    ctx: AutonomyContext,
    wall: Optional[WallSight],
    yaw_deg: Optional[float],
    now: float,
    note: str,
) -> AutonomyOutput:
    """SEARCH 的子阶段动作：advance 直行 → backoff 后退 → slide 沿墙横移。"""
    # 记住出发航向
    if yaw_deg is not None and ctx.heading_ref is None:
        ctx.heading_ref = yaw_deg

    # --- backoff：撞墙后退离墙 ---
    if ctx.search_phase == "backoff":
        if now - ctx.phase_since < SLIDE_BACKOFF_S:
            return AutonomyOutput(BACKOFF_SURGE, 0.0, 0.0, ctx.state, note or "撞墙后退中")
        # 后退完成 → 沿墙横移（切换帧直接走横移动作）
        ctx.search_phase = "slide"
        ctx.phase_since = now
        ctx.slide_target = None
        if yaw_deg is not None and ctx.heading_ref is not None:
            ctx.slide_target = (ctx.heading_ref + 90.0 * ctx.slide_dir) % 360.0
        note = note or "退开了，沿墙横移找球"

    # --- slide：沿墙横移巡逻，单侧超时掉头 ---
    if ctx.search_phase == "slide":
        if ctx.slide_target is None:
            # 无姿态传感器：定时转 1.2 秒再直行，超时掉头
            elapsed = now - ctx.phase_since
            if elapsed < 1.2:
                return AutonomyOutput(
                    0.0, TURN_YAW * ctx.slide_dir, 0.0, ctx.state, note or "转向沿墙方向"
                )
            if elapsed >= SLIDE_MAX_S:
                ctx.slide_dir *= -1
                ctx.phase_since = now
                note = "这侧没找到，掉头去另一侧"
            return AutonomyOutput(SLIDE_SURGE, 0.0, 0.0, ctx.state, note)
        # 有姿态传感器：先转到沿墙航向，再保持航向横移
        diff = _angle_diff(ctx.slide_target, yaw_deg if yaw_deg is not None else 0.0)
        if abs(diff) > TURN_DONE_ERR:
            turn = max(-0.5, min(0.5, TURN_KP * diff))
            return AutonomyOutput(0.0, turn, 0.0, ctx.state, note or "转向沿墙方向")
        if now - ctx.phase_since >= SLIDE_MAX_S:
            ctx.slide_dir *= -1
            ctx.slide_target = (ctx.slide_target + 180.0) % 360.0
            ctx.phase_since = now
            note = "这侧没找到，掉头去另一侧"
        return AutonomyOutput(
            SLIDE_SURGE, _heading_hold(ctx.slide_target, yaw_deg), 0.0, ctx.state, note
        )

    # --- advance：直行找球 ---
    if wall is not None and wall.front >= WALL_FRONT_LIMIT:
        # 到墙边还没见到球（球贴在对面墙边）→ 后退 → 沿墙找
        ctx.search_phase = "backoff"
        ctx.phase_since = now
        return AutonomyOutput(
            BACKOFF_SURGE, 0.0, 0.0, ctx.state,
            note or "到墙边没见到球，先后退再沿墙找",
        )
    yaw_cmd = _heading_hold(ctx.heading_ref, yaw_deg)
    if wall is not None:
        if wall.left >= SIDE_WALL_LIMIT:
            yaw_cmd = max(yaw_cmd, SIDE_FIX_YAW)    # 贴左墙（跑偏）→ 右修
        elif wall.right >= SIDE_WALL_LIMIT:
            yaw_cmd = min(yaw_cmd, -SIDE_FIX_YAW)   # 贴右墙（跑偏）→ 左修
    return AutonomyOutput(SEARCH_SURGE, yaw_cmd, 0.0, ctx.state, note)
