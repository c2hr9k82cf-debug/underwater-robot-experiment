"""rov.autonomy 状态机单元测试（纯逻辑，注入假时间，离线运行）。"""

from __future__ import annotations

from rov.autonomy import (
    ADVANCE_MAX_S,
    APPROACH_SURGE,
    AVOID_YAW,
    BACKOFF_S,
    BACKOFF_SURGE,
    CREEP_S,
    CREEP_SURGE,
    LOST_TIMEOUT_S,
    PEEK_S,
    PEEK_TURN_DEG,
    SEARCH_SURGE,
    SIDE_PEEKS,
    SLIDE_BACKOFF_S,
    TOUCH_S,
    TOUCH_SURGE,
    TOUCH_AREA_RATIO,
    AutonomyContext,
    AutonomyState,
    step,
)
from rov.detector import DetectionResult, WallSight

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

        # SEARCH：无目标时直行前进找球
        out = step(ctx, LOST, FRAME, now)
        assert out.state is AutonomyState.SEARCH
        assert out.surge == SEARCH_SURGE
        assert out.yaw == 0.0

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

    def test_search_goes_straight_after_loss(self) -> None:
        """丢失回搜索后继续直行找球（不转向）。"""
        ctx = AutonomyContext()
        now = 0.0

        step(ctx, det_at(100), FRAME, now)
        step(ctx, det_at(100), FRAME, now)  # ALIGN
        # 丢失超时回 SEARCH
        now += LOST_TIMEOUT_S + 0.1
        out = step(ctx, LOST, FRAME, now)
        assert out.state is AutonomyState.SEARCH
        assert out.surge == SEARCH_SURGE
        assert out.yaw == 0.0


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


class TestWallAvoidance:
    def test_avoid_turns_to_open_side(self) -> None:
        """接近中没看到球、前方被墙占满（墙偏右）：朝空旷的左侧转。"""
        ctx = AutonomyContext()
        step(ctx, det_at(320), FRAME, 0.0)
        step(ctx, det_at(320), FRAME, 0.0)  # → APPROACH
        wall = WallSight(left=0.05, front=0.6, right=0.8)
        out = step(ctx, LOST, FRAME, 0.1, wall=wall)
        assert out.state is AutonomyState.APPROACH
        assert out.surge == 0.0
        assert out.yaw == -AVOID_YAW

    def test_avoid_right_when_left_blocked(self) -> None:
        """墙偏左：朝空旷的右侧转。"""
        ctx = AutonomyContext()
        step(ctx, det_at(320), FRAME, 0.0)
        step(ctx, det_at(320), FRAME, 0.0)  # → APPROACH
        wall = WallSight(left=0.8, front=0.6, right=0.05)
        out = step(ctx, LOST, FRAME, 0.1, wall=wall)
        assert out.yaw == AVOID_YAW

    def test_target_visible_not_avoided(self) -> None:
        """目标可见时直接冲目标，不避障（球贴墙场景必须能碰到球）。"""
        ctx = AutonomyContext()
        step(ctx, det_at(320), FRAME, 0.0)
        step(ctx, det_at(320), FRAME, 0.0)  # → APPROACH
        wall = WallSight(left=0.1, front=0.6, right=0.1)  # 球背后是墙
        out = step(ctx, det_at(320), FRAME, 0.0, wall=wall)
        assert out.surge == APPROACH_SURGE

    def test_backoff_not_interrupted(self) -> None:
        """后退是避险动作，不被避障打断。"""
        ctx = AutonomyContext()
        ctx.state = AutonomyState.BACKOFF
        wall = WallSight(left=0.9, front=0.9, right=0.9)
        out = step(ctx, LOST, FRAME, 0.0, wall=wall)
        assert out.surge == BACKOFF_SURGE


class TestWallSearch:
    """撞墙 → 持续后退离墙 → 小角度转停看 → 小步挪（长方形水池对侧贴墙找球）。"""

    def test_hit_wall_backs_off(self) -> None:
        """直行到墙没见球：先后退。"""
        ctx = AutonomyContext()
        wall = WallSight(left=0.2, front=0.5, right=0.3)
        out = step(ctx, LOST, FRAME, 0.0, wall=wall)
        assert out.state is AutonomyState.SEARCH
        assert out.surge < 0
        assert ctx.search_phase == "backoff"

    def test_backoff_continues_until_wall_clear(self) -> None:
        """避障不是只做一次：墙没退开就一直后退，退开才进入停看。"""
        ctx = AutonomyContext()
        wall = WallSight(0.2, 0.5, 0.3)
        step(ctx, LOST, FRAME, 0.0, wall=wall)              # → backoff
        # 5 秒后墙还在 → 继续后退（不是只退一次）
        out = step(ctx, LOST, FRAME, SLIDE_BACKOFF_S + 5.0, wall=wall)
        assert ctx.search_phase == "backoff"
        assert out.surge == BACKOFF_SURGE
        # 墙退开了 → 转入停看
        out2 = step(ctx, LOST, FRAME, SLIDE_BACKOFF_S + 6.0, wall=WallSight(0.1, 0.1, 0.1))
        assert ctx.search_phase == "peek"
        assert out2.surge == 0.0

    def test_wall_recheck_in_any_phase(self) -> None:
        """任何子阶段前方出现墙都立即回后退（不是一次性的）。"""
        ctx = AutonomyContext()
        ctx.search_phase = "creep"
        ctx.phase_since = 0.0
        wall = WallSight(0.3, 0.5, 0.3)
        out = step(ctx, LOST, FRAME, 0.5, wall=wall)
        assert ctx.search_phase == "backoff"
        assert out.surge == BACKOFF_SURGE

    def test_backoff_then_peek_small_turn(self) -> None:
        """后退完成进入停看：小角度转（不是 90°）。"""
        ctx = AutonomyContext()
        wall = WallSight(0.2, 0.5, 0.3)
        step(ctx, LOST, FRAME, 0.0, wall=wall)          # → backoff
        out = step(ctx, LOST, FRAME, SLIDE_BACKOFF_S, wall=WallSight(0.1, 0.1, 0.1))
        assert ctx.search_phase == "peek"
        assert out.yaw != 0.0                            # 无姿态 fallback：转向

    def test_peek_with_yaw_turns_then_stops(self) -> None:
        """有姿态传感器：转到小角度目标（25°），到位后停桨静止观察。"""
        ctx = AutonomyContext()
        wall = WallSight(0.2, 0.5, 0.3)
        step(ctx, LOST, FRAME, 0.0, wall=wall, yaw_deg=0.0)          # 记住航向 0°
        out = step(ctx, LOST, FRAME, SLIDE_BACKOFF_S, wall=WallSight(0.1, 0.1, 0.1),
                   yaw_deg=0.0)
        assert ctx.search_phase == "peek"
        assert ctx.slide_target == PEEK_TURN_DEG        # 小角度 25°，不是 90°
        assert out.yaw > 0                              # 右转到 25°
        # 转到位（25°）后 → 停桨静止观察（"歇着看他"）
        out2 = step(ctx, LOST, FRAME, SLIDE_BACKOFF_S + 1.0, yaw_deg=25.0)
        assert out2.surge == 0.0 and out2.yaw == 0.0

    def test_peek_timeout_creeps_then_peeks_again(self) -> None:
        """停看没找到 → 小步挪 → 再转一点再停看（角度递增）。"""
        ctx = AutonomyContext()
        ctx.search_phase = "peek"
        ctx.peek_turn_done = True
        ctx.phase_since = 0.0
        ctx.heading_ref = 0.0
        ctx.slide_target = PEEK_TURN_DEG
        # 静止观察时长到 → 进 creep
        out = step(ctx, LOST, FRAME, PEEK_S + 0.1, yaw_deg=PEEK_TURN_DEG)
        assert ctx.search_phase == "creep"
        assert out.surge == CREEP_SURGE
        # creep 时长到 → 回 peek，目标角度递增到 50°
        out2 = step(ctx, LOST, FRAME, PEEK_S + 0.1 + CREEP_S + 0.1,
                    yaw_deg=PEEK_TURN_DEG)
        assert ctx.search_phase == "peek"
        assert ctx.slide_target == PEEK_TURN_DEG * 2     # 逐次递增
        assert out2.surge == 0.0                         # 又停下来看

    def test_reverses_after_side_peeks(self) -> None:
        """同侧停看满 3 次：掉头去另一侧。"""
        ctx = AutonomyContext()
        ctx.search_phase = "creep"
        ctx.phase_since = 0.0
        ctx.peek_count = SIDE_PEEKS - 1
        ctx.slide_dir = 1.0
        step(ctx, LOST, FRAME, CREEP_S + 0.1, yaw_deg=0.0)
        assert ctx.slide_dir == -1.0
        assert ctx.peek_count == 0

    def test_advance_timeout_fallback(self) -> None:
        """兜底：直行太久没见到球（可能墙检测漏判）→ 后退换方向，不许一直直行。"""
        ctx = AutonomyContext()
        ctx.state_since = 0.0
        out = step(ctx, LOST, FRAME, ADVANCE_MAX_S + 1.0)
        assert ctx.search_phase == "backoff"
        assert out.surge == BACKOFF_SURGE

    def test_side_wall_correction(self) -> None:
        """直行时贴左墙（跑偏）：视觉居中向池心右修（无姿态数据也生效）。"""
        ctx = AutonomyContext()
        wall = WallSight(left=0.6, front=0.1, right=0.1)
        out = step(ctx, LOST, FRAME, 0.0, wall=wall)
        assert out.yaw > 0

    def test_heading_hold_corrects_drift(self) -> None:
        """航向保持：出发航向 0°，漂到 -20° → 右修回来。"""
        ctx = AutonomyContext()
        step(ctx, LOST, FRAME, 0.0, yaw_deg=0.0)          # 记住航向
        out = step(ctx, LOST, FRAME, 0.1, yaw_deg=-20.0)
        assert out.yaw > 0
        assert out.surge == SEARCH_SURGE
