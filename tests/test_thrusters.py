"""rov.thrusters 单元测试（纯函数，离线运行）。"""

from __future__ import annotations

import pytest

from rov.thrusters import (
    PWM_MAX,
    PWM_MIN,
    PWM_NEUTRAL,
    Attitude,
    build_command,
    clamp_pwm,
    mix,
    parse_sensor,
    shutdown_command,
    stop_command,
)


class TestClamp:
    def test_clamp_limits(self) -> None:
        assert clamp_pwm(0) == PWM_MIN
        assert clamp_pwm(9999) == PWM_MAX
        assert clamp_pwm(PWM_NEUTRAL) == PWM_NEUTRAL

    def test_clamp_keeps_valid_value(self) -> None:
        assert clamp_pwm(1523) == 1523


class TestMix:
    def test_stop_is_neutral(self) -> None:
        """零推力 → 4 路全部中位。"""
        pwm = mix(0.0, 0.0, 0.0)
        assert pwm.as_tuple() == (1462, 1462, 1462, 1462)

    def test_pdf_forward_example(self) -> None:
        """PDF 示例：前进时左 1523、右 1401（反桨对称）。"""
        pwm = mix(0.305, 0.0, 0.0)
        assert pwm.left == 1523
        assert pwm.right == 1401

    def test_backward_is_symmetric(self) -> None:
        fwd = mix(0.5, 0.0, 0.0)
        back = mix(-0.5, 0.0, 0.0)
        # 后退是前进的左右互换（反桨）
        assert back.left == 2 * PWM_NEUTRAL - fwd.left
        assert back.right == 2 * PWM_NEUTRAL - fwd.right

    def test_pure_yaw_same_offset(self) -> None:
        """纯转向：左右同号偏移（反桨产生力矩）。"""
        pwm = mix(0.0, 0.5, 0.0)
        assert pwm.left > PWM_NEUTRAL
        assert pwm.right > PWM_NEUTRAL
        assert pwm.left == pwm.right

    def test_yaw_signs(self) -> None:
        right_turn = mix(0.0, 0.5, 0.0)
        left_turn = mix(0.0, -0.5, 0.0)
        assert right_turn.left > left_turn.left

    def test_heave_no_reversal(self) -> None:
        """竖直推进器不反向：上浮双正，下潜双负。"""
        up = mix(0.0, 0.0, 1.0)
        down = mix(0.0, 0.0, -1.0)
        assert up.top_left == PWM_MAX and up.top_right == PWM_MAX
        assert down.top_left == PWM_MIN and down.top_right == PWM_MIN

    def test_thrust_clamped(self) -> None:
        """推力超界自动限幅，不超 PWM 范围。"""
        pwm = mix(5.0, -5.0, 3.0)
        for v in pwm.as_tuple():
            assert PWM_MIN <= v <= PWM_MAX

    def test_combined_axes(self) -> None:
        """多轴叠加：前进+右转，左推进器偏移更大。"""
        pwm = mix(0.4, 0.3, 0.0)
        assert pwm.left > pwm.right
        assert pwm.left > PWM_NEUTRAL


class TestCommand:
    def test_format(self) -> None:
        pwm = mix(0.305, 0.0, 0.0)
        assert build_command(pwm) == "1523:1401:1462:1462:0"

    def test_shutdown_flag(self) -> None:
        assert shutdown_command().endswith(":1")
        assert stop_command().endswith(":0")

    def test_stop_command_is_neutral(self) -> None:
        assert stop_command() == "1462:1462:1462:1462:0"


class TestParseSensor:
    def test_normal(self) -> None:
        att = parse_sensor("0:1:180")
        assert att == Attitude(roll=0.0, pitch=1.0, yaw=180.0)

    def test_negative_values(self) -> None:
        att = parse_sensor("-12.5:3.0:-90")
        assert att is not None
        assert att.roll == pytest.approx(-12.5)

    def test_bad_format(self) -> None:
        assert parse_sensor("10:20") is None
        assert parse_sensor("a:b:c") is None
        assert parse_sensor("") is None
