"""rov.autonomy 状态机单元测试（纯逻辑，注入假时间，离线运行）。"""

from __future__ import annotations

from rov.autonomy import (
    APPROACH_SURGE,
    BACKOFF_S,
    BACKOFF_SURGE,
    LOST_TIMEOUT_S,
    SEARCH_YAW,
    TOUCH_S,
    TOUCH_SURGE,
    TOUCH_AREA_RATIO,
    AutonomyContext,
    AutonomyState,
    step,
)
from rov.detector import DetectionResult

FRAME = (640, 480)  # 宽, 高
FRAME_AREA = 640 * 480


def det_at(cx: float, cy: float = 240.0, area: float | None = None) -> DetectionResult:
    """构造一个 DETECTED 结果。area 默认占画面 2%。"""
    if area is None:
        area = FRAME_AREA * 0.02
    return DetectionResult.detected(
        center=(cx, cy), area=area, bbox=(int(cx - 20), int(cy - 20), 40, 40), radius=20.0
    )


LOST = DetectionResult.lost()


class TestSearchToDone:
    def test_full_happy_path(self) -> None:
        """找球→对准→接近→轻触→后退→完成 全链路。"""
        ctx = AutonomyContext()
        now = 0.0

        # SEARCH：无目标时原地旋转
        out = step(ctx, LOST, FRAME, now)
        assert out.state is AutonomyState.SEARCH
        assert abs(out.yaw) == SEARCH_YAW

        # 检出（居中）→ ALIGN
        out = step(ctx, det_at(320), FRAME, now)
        assert out.state is AutonomyState.ALIGN

        # ALIGN 居中误差 0 → APPROACH
        out = step(ctx, det_at(320), FRAME, now)
        assert out.state is AutonomyState.APPROACH

        # APPROACH 前进
        out = step(ctx, det_at(320), FRAME, now)
        assert out.state is AutonomyState.APPROACH
        assert out.surge == APPROACH_SURGE

        # 目标够大 → TOUCH
        big = det_at(320, area=FRAME_AREA * (TOUCH_AREA_RATIO + 0.02))
        out = step(ctx, big, FRAME, now)
        assert out.state is AutonomyState.TOUCH
        assert out.surge == TOUCH_SURGE

        # TOUCH 时长到 → BACKOFF
        now += TOUCH_S
        out = step(ctx, big, FRAME, now)
        assert out.state is AutonomyState.BACKOFF
        assert out.surge == BACKOFF_SURGE

        # BACKOFF 时长到 → DONE
        now += BACKOFF_S
        out = step(ctx, big, FRAME, now)
        assert out.state is AutonomyState.DONE
        assert ctx.done

        # DONE 保持停桨
        out = step(ctx, det_at(320), FRAME, now + 10)
        assert out.state is AutonomyState.DONE
        assert out.surge == 0.0 and out.yaw == 0.0

    def test_search_yaw_follows_last_seen(self) -> None:
        """回搜索后朝最后看到球的方向转。"""
        ctx = AutonomyContext()
        now = 0.0

        # 检出在画面左侧（cx < 320）
        step(ctx, det_at(100), FRAME, now)
        step(ctx, det_at(100), FRAME, now)  # ALIGN
        # 丢失超时回 SEARCH
        now += LOST_TIMEOUT_S + 0.1
        out = step(ctx, LOST, FRAME, now)
        assert out.state is AutonomyState.SEARCH
        assert out.yaw < 0  # 左侧 → 左转找


class TestAlign:
    def test_yaw_sign_matches_error(self) -> None:
        """球在右 → 右转（yaw>0）；球在左 → 左转（yaw<0）。"""
        ctx = AutonomyContext()
        step(ctx, det_at(320), FRAME, 0.0)  # 进 ALIGN

        out = step(ctx, det_at(520), FRAME, 0.0)  # 球在右
        assert out.state is AutonomyState.ALIGN
        assert out.yaw > 0

        out = step(ctx, det_at(120), FRAME, 0.0)  # 球在左
        assert out.yaw < 0

    def test_no_forward_in_align(self) -> None:
        ctx = AutonomyContext()
        step(ctx, det_at(320), FRAME, 0.0)
        out = step(ctx, det_at(520), FRAME, 0.0)
        assert out.surge == 0.0


class TestLostRecovery:
    def test_brief_loss_keeps_state(self) -> None:
        """短暂丢失保持当前状态，不乱转。"""
        ctx = AutonomyContext()
        step(ctx, det_at(320), FRAME, 0.0)          # → ALIGN
        step(ctx, det_at(520), FRAME, 0.0)          # ALIGN 中
        out = step(ctx, LOST, FRAME, 0.1)           # 刚丢 0.1s
        assert out.state is AutonomyState.ALIGN
        assert out.yaw == 0.0

    def test_long_loss_returns_search(self) -> None:
        """丢失超过阈值回 SEARCH。"""
        ctx = AutonomyContext()
        step(ctx, det_at(320), FRAME, 0.0)
        step(ctx, det_at(520), FRAME, 0.0)
        now = LOST_TIMEOUT_S + 0.2
        out = step(ctx, LOST, FRAME, now)
        assert out.state is AutonomyState.SEARCH

    def test_loss_during_approach_returns_search(self) -> None:
        """接近中丢失同样回搜索。"""
        ctx = AutonomyContext()
        step(ctx, det_at(320), FRAME, 0.0)
        step(ctx, det_at(320), FRAME, 0.0)  # → APPROACH
        step(ctx, det_at(320), FRAME, 0.0)
        now = LOST_TIMEOUT_S + 0.2
        out = step(ctx, LOST, FRAME, now)
        assert out.state is AutonomyState.SEARCH


class TestLowConfidence:
    def test_low_conf_only_aligns(self) -> None:
        """低置信（目标极小）只对中不前进，防浑水误吸附。"""
        ctx = AutonomyContext()
        step(ctx, det_at(320), FRAME, 0.0)
        step(ctx, det_at(320), FRAME, 0.0)  # → APPROACH

        tiny = det_at(500, area=FRAME_AREA * 0.002)  # 远小于阈值
        out = step(ctx, tiny, FRAME, 0.0)
        assert out.state is AutonomyState.APPROACH
        assert out.surge == 0.0      # 不前进
        assert out.yaw > 0            # 但会转向目标
