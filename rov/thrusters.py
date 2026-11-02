"""推进器协议与推力映射模块。

职责（纯函数，无 IO）：
- 推力指令 [-1, 1] → 4 路推进器 PWM（1262-1662，中位 1462）
- 组装控制指令字符串（5 段冒号分隔）
- 解析传感器回传（roll:pitch:yaw）

协议要点（机器人使用说明.pdf）：
- 水平推进器正反桨叶布置，同一 PWM 对左右推进器产生反向推力。
  前进示例：左 1523（1462+61）、右 1401（1462-61）。
- 竖直推进器不需要反向，同号 PWM 同向推力。
- 推进器长时间不收指令会鸣叫，client 需持续发送（见 rov/comm.py）。
- 关机标志：0=正常工作，1=关机（慎用，仅显式请求时发送）。
"""

from __future__ import annotations

from dataclasses import dataclass

# --- PWM 标定常量（来自机器人使用说明） ---
PWM_MIN: int = 1262
PWM_MAX: int = 1662
PWM_NEUTRAL: int = 1462          # 停桨中位
PWM_RANGE: int = PWM_MAX - PWM_NEUTRAL  # 单侧行程 200

# 推力指令合法范围
THRUST_MIN: float = -1.0
THRUST_MAX: float = 1.0


def clamp_pwm(value: float) -> int:
    """把 PWM 值限幅到 [PWM_MIN, PWM_MAX] 并取整。"""
    return int(round(min(max(value, PWM_MIN), PWM_MAX)))


def clamp_thrust(value: float) -> float:
    """把推力值限幅到 [-1, 1]。"""
    return min(max(value, THRUST_MIN), THRUST_MAX)


@dataclass(frozen=True)
class ThrusterPWM:
    """4 路推进器 PWM 值。"""

    left: int        # 左推进器
    right: int       # 右推进器
    top_left: int    # 左上（竖直）推进器
    top_right: int   # 右上（竖直）推进器

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.right, self.top_left, self.top_right)


def mix(surge: float, yaw: float, heave: float) -> ThrusterPWM:
    """把推力指令混控为 4 路 PWM。

    参数（各 ∈ [-1, 1]，超界自动限幅）：
    - surge: 前后，正=前进，负=后退
    - yaw:   转向，正=右转，负=左转
    - heave: 升沉，正=上浮，负=下潜

    反桨混控：
    - 左推进器   PWM = 1462 + (surge + yaw) * 200
    - 右推进器   PWM = 1462 - (surge - yaw) * 200   （反桨，符号对调）
    - 竖直推进器 PWM = 1462 + heave * 200            （不反向）

    校验（PDF 示例）：surge=0.305, yaw=0 → 左 1523 / 右 1401。
    """
    surge = clamp_thrust(surge)
    yaw = clamp_thrust(yaw)
    heave = clamp_thrust(heave)

    left = clamp_pwm(PWM_NEUTRAL + (surge + yaw) * PWM_RANGE)
    right = clamp_pwm(PWM_NEUTRAL - (surge - yaw) * PWM_RANGE)
    top = clamp_pwm(PWM_NEUTRAL + heave * PWM_RANGE)

    return ThrusterPWM(left=left, right=right, top_left=top, top_right=top)


def build_command(pwm: ThrusterPWM, *, shutdown: bool = False) -> str:
    """组装控制指令字符串：左:右:左上:右上:关机标志。"""
    flag = 1 if shutdown else 0
    return f"{pwm.left}:{pwm.right}:{pwm.top_left}:{pwm.top_right}:{flag}"


def stop_command() -> str:
    """中位停桨指令（正常模式）。"""
    return build_command(mix(0.0, 0.0, 0.0))


def shutdown_command() -> str:
    """关机指令（flag=1）。仅在用户显式请求时使用。"""
    return build_command(mix(0.0, 0.0, 0.0), shutdown=True)


@dataclass(frozen=True)
class Attitude:
    """传感器姿态（单位：度）。"""

    roll: float
    pitch: float
    yaw: float


def parse_sensor(message: str) -> Attitude | None:
    """解析传感器回传字符串 "roll:pitch:yaw"。

    格式不对或非数值时返回 None（不抛异常）。
    """
    parts = message.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        roll, pitch, yaw = (float(p) for p in parts)
    except ValueError:
        return None
    return Attitude(roll=roll, pitch=pitch, yaw=yaw)
