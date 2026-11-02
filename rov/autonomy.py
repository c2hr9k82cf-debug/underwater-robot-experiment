"""自主运动状态机（纯逻辑，无 IO）。

状态与转换：
    SEARCH    慱速旋转找球            ──检出──▶ ALIGN
    ALIGN     把球心转到画面中央      ──对中──▶ APPROACH
    APPROACH  前进接近 + 持续对中修正  ──够大──▶ TOUCH
                                 └──丢失>0.8s──▶ SEARCH（朝最后方向转）
    TOUCH     低速轻推 1.5s          ──▶ BACKOFF
    BACKOFF   后退 1.5s              ──▶ DONE
    DONE      停桨，任务结束

浑水中目标频繁丢失是常态，所以：
- 丢失恢复带方向记忆（朝最后看到球的方向先转）
- 低置信检测只对中不前进，防止在浑水误检上"撞山"

决策函数是纯函数（输入检测结果+时间 → 输出推力+新状态），
全部可离线单测。本模块不含任何通信/显示代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from rov.detector import DetectionResult

# --- 可调参数 ---
SEARCH_YAW: float = 0.25          # 搜索旋转速度
ALIGN_KP: float = 0.8             # 对中 P 增益（误差→转向角速度）
ALIGN_DONE_ERR: float = 0.12      # |归一化横偏差| 小于该值视为对中
APPROACH_SURGE: float = 0.30      # 接近前进速度
APPROACH_YAW_CAP: float = 0.35    # 接近时对中修正上限
TOUCH_AREA_RATIO: float = 0.12    # 目标面积占画面比例达到该值→轻触
TOUCH_SURGE: float = 0.20         # 轻推速度
TOUCH_S: float = 1.5              # 轻推时长
BACKOFF_SURGE: float = -0.25      # 后退速度
BACKOFF_S: float = 1.5            # 后退时长
LOST_TIMEOUT_S: float = 0.8       # 丢失超过该时长回搜索
LOW_CONF_AREA_RATIO: float = 0.005  # 面积占比低于此视为低置信（只对中不前进）


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


def _norm_cx(result: DetectionResult, frame_w: int) -> float:
    """归一化横偏差 ∈ [-1, 1]，正=球在画面右侧。"""
    assert result.center is not None
    return (result.center[0] - frame_w / 2.0) / (frame_w / 2.0)


def step(
    ctx: AutonomyContext,
    result: DetectionResult,
    frame_size: Tuple[int, int],
    now: float,
) -> AutonomyOutput:
    """状态机单步决策（ctx 是唯一可变记忆）。

    每帧先做一次状态切换（最多一次），然后输出**新状态**的动作，
    切换帧不停顿。参数：
    - ctx: 跨帧记忆（调用方持有并复用）
    - result: 本帧检测结果
    - frame_size: (宽, 高)
    - now: 单调时钟秒（测试可注入假时间）
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
        note = "发现目标，开始对准"
    elif ctx.state in (AutonomyState.ALIGN, AutonomyState.APPROACH) and not detected \
            and lost_for > LOST_TIMEOUT_S:
        ctx.state = AutonomyState.SEARCH
        ctx.state_since = now
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

    # ---- 当前状态动作 ----
    if ctx.state is AutonomyState.SEARCH:
        # 朝最后看到球的方向转；从未见过就默认右转
        direction = 1.0
        if ctx.last_seen_cx is not None:
            direction = 1.0 if ctx.last_seen_cx > 0 else -1.0
        return AutonomyOutput(0.0, direction * SEARCH_YAW, 0.0, ctx.state, note)

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
