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
    advance   直行前进找球（航向保持 + 视觉居中防跑偏）
    backoff   前方有墙 → 后退离墙（墙没退开就一直退，不止退一次）
    peek      小角度转（25° 逐次递增）→ 停桨静止观察找球
    creep     没找到就小步挪一下 → 再转一点再停看
    同侧看满 3 次（累计约 75°）掉头看另一侧，来回直到发现球。
    子阶段是 SEARCH 内部记忆，检出即转入 ALIGN；目标丢失回 SEARCH 从 advance 重新开始。

避障每帧都查、不是一次性：任何搜索子阶段前方出现墙都立即回 backoff 后退；
直行超过 ADVANCE_MAX_S 未见球也兜底后退换方向（防墙检测漏判一直撞墙）。

姿态传感器（yaw）用法：记住出发航向 → 直行保持航向防跑偏；
停看小角度转按航向精确定向。无姿态数据时用左右走廊视觉居中 + 定时转向兜底。

决策函数是纯函数（输入检测结果+墙壁+姿态+时间 → 输出推力+新状态），
全部可离线单测。本模块不含任何通信/显示代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from rov.config import WALL_FRONT_LIMIT
from rov.detector import DetectionResult, WallSight

# --- 可调参数（实机反馈：转向要转得动、推进要游得动，数值偏大取）---
SEARCH_SURGE: float = 0.7         # 搜索直行前进速度
AVOID_YAW: float = 0.7            # 避障转向角速度
ALIGN_KP: float = 0.8             # 对中 P 增益（误差→转向角速度）
ALIGN_DONE_ERR: float = 0.12      # |归一化横偏差| 小于该值视为对中
ALIGN_YAW_CAP: float = 0.8        # 对中转向上限
APPROACH_SURGE: float = 0.7       # 接近前进速度
APPROACH_YAW_CAP: float = 0.5     # 接近时对中修正上限
TOUCH_AREA_RATIO: float = 0.12    # 目标面积占画面比例达到该值→轻触
TOUCH_SURGE: float = 0.5          # 轻推速度
TOUCH_S: float = 1.5              # 轻推时长
BACKOFF_SURGE: float = -0.6       # 后退速度
BACKOFF_S: float = 1.5            # 后退时长
LOST_TIMEOUT_S: float = 0.8       # 丢失超过该时长回搜索
LOW_CONF_AREA_RATIO: float = 0.005  # 面积占比低于此视为低置信（只对中不前进）

# --- 沿墙搜索（撞墙后退 → 小角度转停看 → 小步挪）---
SLIDE_BACKOFF_S: float = 1.5      # 撞墙后至少后退时长（墙没退开则持续后退）
PEEK_TURN_DEG: float = 25.0       # 每次停看的小角度转向（逐次递增，不转 90°）
PEEK_TURN_TIME: float = 0.7       # 无姿态传感器时的转向时长（秒）
PEEK_S: float = 2.0               # 停桨静止观察时长（画面稳，检测容易）
CREEP_S: float = 2.0              # 小步挪动时长
CREEP_SURGE: float = 0.5          # 小步挪动速度
SIDE_PEEKS: int = 3               # 同侧停看次数，看满掉头去另一侧
ADVANCE_MAX_S: float = 30.0       # 直行兜底：超时未见球也后退换方向
TURN_YAW: float = 0.8             # 转向速度（无姿态传感器 fallback）
TURN_KP: float = 0.03             # 转向航向 P 增益（度误差→推力，25°→0.75）
TURN_CAP: float = 1.0             # 转向推力上限（给足，否则水里转不动）
TURN_DONE_ERR: float = 6.0        # 转向到位角度容差（度）
HEADING_KP: float = 0.05          # 航向保持增益（度误差→推力）
HEADING_CAP: float = 0.5          # 航向/居中修正上限
CENTER_FIX_KP: float = 0.5        # 视觉居中增益（左右走廊蓝色占比差→转向）


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
    state_since: float = -1.0           # 进入当前状态的时刻（-1=未初始化）
    last_seen_at: float = -1.0          # 最后一次检出时刻
    last_seen_cx: Optional[float] = None  # 最后检出的归一化横坐标
    done: bool = False
    # --- SEARCH 沿墙搜索子阶段 ---
    search_phase: str = "advance"       # advance / backoff / peek / creep
    phase_since: float = -1.0           # 进入当前子步骤的时刻（-1=未初始化）
    heading_ref: Optional[float] = None  # 出发航向（度）
    slide_target: Optional[float] = None  # 本次停看的目标航向（度）
    slide_dir: float = 1.0              # 停看方向 +1 / -1
    peek_turn_done: bool = False        # 停看前的小角度转是否到位
    peek_count: int = 0                 # 同侧已停看次数（超 SIDE_PEEKS 掉头）


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


def _peek_target(ctx: AutonomyContext, yaw_deg: Optional[float]) -> Optional[float]:
    """本次停看的目标航向：出发航向 + 递增小角度（无姿态数据返回 None）。"""
    if yaw_deg is None or ctx.heading_ref is None:
        return None
    step_deg = PEEK_TURN_DEG * (ctx.peek_count + 1) * ctx.slide_dir
    return (ctx.heading_ref + step_deg) % 360.0


def _reset_search(ctx: AutonomyContext, now: float) -> None:
    """回到直行搜索初始状态（保留 slide_dir 历史）。"""
    ctx.search_phase = "advance"
    ctx.phase_since = now
    ctx.heading_ref = None
    ctx.slide_target = None
    ctx.peek_turn_done = False
    ctx.peek_count = 0


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

    # 时间戳未初始化：真实单调时钟是大数，绝不能拿 0.0 当"起点"
    # （否则开局第一帧就被超时兜底误判成"直行太久"）
    if ctx.state_since < 0:
        ctx.state_since = now
    if ctx.phase_since < 0:
        ctx.phase_since = now

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
        _reset_search(ctx, now)
        note = "发现目标，开始对准"
    elif ctx.state in (AutonomyState.ALIGN, AutonomyState.APPROACH) and not detected \
            and lost_for > LOST_TIMEOUT_S:
        ctx.state = AutonomyState.SEARCH
        ctx.state_since = now
        _reset_search(ctx, now)
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
    # SEARCH 的墙处理由下面的沿墙搜索子阶段负责（每帧都查）；ALIGN 不前进；BACKOFF 在后退。
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
        yaw = max(-ALIGN_YAW_CAP, min(ALIGN_YAW_CAP, ALIGN_KP * err))
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
    """SEARCH 子阶段动作：advance 直行 → backoff 后退 → peek 停看 → creep 小步挪。"""
    # 记住出发航向
    if yaw_deg is not None and ctx.heading_ref is None:
        ctx.heading_ref = yaw_deg

    # ---- 避障不是一次性：任何子阶段前方有墙都退回后退（不止退一次）----
    if wall is not None and wall.front >= WALL_FRONT_LIMIT \
            and ctx.search_phase != "backoff":
        ctx.search_phase = "backoff"
        ctx.phase_since = now
        note = note or "前方有墙，先后退"

    # --- backoff → peek → creep 子阶段（切换后重新评估，切换帧不停顿）---
    while True:
        # backoff：后退离墙；墙没退开就一直退
        if ctx.search_phase == "backoff":
            still_blocked = wall is not None and wall.front >= WALL_FRONT_LIMIT
            if still_blocked or (now - ctx.phase_since < SLIDE_BACKOFF_S):
                return AutonomyOutput(BACKOFF_SURGE, 0.0, 0.0, ctx.state, note or "后退离墙中")
            # 退开了 → 小角度转，停下来找球
            ctx.search_phase = "peek"
            ctx.peek_turn_done = False
            ctx.phase_since = now
            ctx.slide_target = _peek_target(ctx, yaw_deg)
            note = note or "退开了，转一点停下来找球"
            continue

        # peek：小角度转 → 停桨静止观察
        if ctx.search_phase == "peek":
            if not ctx.peek_turn_done:
                if ctx.slide_target is not None and yaw_deg is not None:
                    diff = _angle_diff(ctx.slide_target, yaw_deg)
                    if abs(diff) > TURN_DONE_ERR:
                        turn = max(-TURN_CAP, min(TURN_CAP, TURN_KP * diff))
                        return AutonomyOutput(0.0, turn, 0.0, ctx.state, note or "小角度转向")
                elif now - ctx.phase_since < PEEK_TURN_TIME:
                    return AutonomyOutput(
                        0.0, TURN_YAW * ctx.slide_dir, 0.0, ctx.state, note or "小角度转向"
                    )
                # 转到位 → 静止观察计时开始
                ctx.peek_turn_done = True
                ctx.phase_since = now
            if now - ctx.phase_since < PEEK_S:
                return AutonomyOutput(0.0, 0.0, 0.0, ctx.state, note or "停下来观察找球")
            # 这里没有 → 小步挪一下再看
            ctx.search_phase = "creep"
            ctx.phase_since = now
            note = note or "这里没有，小步挪一下再看"
            continue

        # creep：沿当前朝向小步挪动
        if ctx.search_phase == "creep":
            if now - ctx.phase_since < CREEP_S:
                return AutonomyOutput(
                    CREEP_SURGE, _heading_hold(ctx.slide_target, yaw_deg), 0.0, ctx.state, note
                )
            # 挪完 → 再转一点停看；同侧看满掉头去另一侧
            ctx.peek_count += 1
            if ctx.peek_count >= SIDE_PEEKS:
                ctx.peek_count = 0
                ctx.slide_dir *= -1
                note = "这侧没找到，掉头去另一侧"
            else:
                note = note or "再转一点停下来找"
            ctx.search_phase = "peek"
            ctx.peek_turn_done = False
            ctx.phase_since = now
            ctx.slide_target = _peek_target(ctx, yaw_deg)
            continue

        break  # advance（子阶段之外）

    # --- advance：直行找球（航向保持 + 视觉居中）---
    if now - ctx.state_since >= ADVANCE_MAX_S:
        # 兜底：直行太久没见到球（可能墙检测漏判）→ 后退换方向
        ctx.search_phase = "backoff"
        ctx.phase_since = now
        return AutonomyOutput(
            BACKOFF_SURGE, 0.0, 0.0, ctx.state, "直行太久没找到，后退换方向"
        )
    yaw_cmd = _heading_hold(ctx.heading_ref, yaw_deg)
    if wall is not None:
        # 视觉居中：贴左墙（left 大）→ 右修；贴右墙 → 左修；无姿态数据也能走直
        yaw_cmd += CENTER_FIX_KP * (wall.left - wall.right)
    yaw_cmd = max(-HEADING_CAP, min(HEADING_CAP, yaw_cmd))
    return AutonomyOutput(SEARCH_SURGE, yaw_cmd, 0.0, ctx.state, note)
