"""模式运行时公共组件。

- CommRunner：把 RobotComm 的 asyncio 事件循环放到后台线程，
  视频/键盘主循环（cv2，同步）可以安全调用 set_thrust / stop
- draw_hud：在画面左上角画半透明状态面板（终端不刷屏，状态看画面）
- 其他小工具（确认提示、占位帧）
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from rov.comm import RobotComm
from rov.thrusters import Attitude, build_command, mix, parse_sensor  # noqa: F401

# --- HUD 样式 ---
HUD_ORIGIN = (10, 10)
HUD_LINE_H = 22
HUD_PAD = 8
HUD_ALPHA = 0.55
HUD_TEXT_COLOR = (255, 255, 255)
HUD_BG_COLOR = (30, 30, 30)


class CommRunner:
    """在后台线程运行 RobotComm，供同步 UI 主循环调用。"""

    def __init__(self, url: str, live: bool) -> None:
        self.comm = RobotComm(url, live=live)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        ready = threading.Event()

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self.comm.start())
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        if not ready.wait(timeout=5):
            raise RuntimeError("通信线程启动超时")

    def set_thrust(self, surge: float, yaw: float, heave: float) -> None:
        self.comm.set_thrust(surge, yaw, heave)

    def stop(self) -> None:
        """停止通信并安全收尾（补发中位停桨帧）。"""
        if self._loop is None:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self.comm.stop(), self._loop)
            future.result(timeout=5)
        except Exception:  # noqa: BLE001 - 收尾失败不阻塞退出
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    # --- 状态读取（UI 线程直接读，float 赋值原子） ---

    @property
    def connected(self) -> bool:
        return self.comm.connected

    @property
    def last_command(self) -> str:
        return self.comm.last_command

    @property
    def attitude(self) -> Optional[Attitude]:
        return self.comm.latest_attitude


def draw_hud(frame: np.ndarray, lines: List[str]) -> None:
    """在画面左上角画半透明 HUD 面板（原地绘制）。"""
    if not lines:
        return
    w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for s in lines)
    h = HUD_LINE_H * len(lines)
    x, y = HUD_ORIGIN
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - HUD_PAD // 2, y - HUD_PAD // 2),
        (x + w + HUD_PAD, y + h + HUD_PAD // 2),
        HUD_BG_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (x, y + HUD_LINE_H * (i + 1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_TEXT_COLOR, 1, cv2.LINE_AA,
        )


def placeholder_frame(message: str = "视频断开，重连中...") -> np.ndarray:
    """视频不可用时的占位画面。"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame, message, (60, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA,
    )
    return frame


def confirm(prompt: str) -> bool:
    """终端 y/N 确认（危险操作前必须调用）。"""
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in ("y", "yes")


# 截图保存目录（项目根下，运行时生成，不入库）
SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "snapshots"


def save_snapshot(frame: np.ndarray) -> Path:
    """保存当前帧**原图**（不带标注）到 snapshots/，返回文件路径。

    存原图是为了之后能用 analyze_color.py 重新标定；
    文件名带时间戳，同秒多次保存自动加序号。
    """
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"snap_{stamp}.jpg"
    n = 1
    while path.exists():
        path = SNAPSHOT_DIR / f"snap_{stamp}_{n}.jpg"
        n += 1
    cv2.imwrite(str(path), frame)
    return path


def pwm_hud_line(command: str) -> str:
    """把指令字符串变成 HUD 显示行。"""
    parts = command.split(":")
    if len(parts) != 5:
        return f"指令: {command}"
    return (
        f"PWM L={parts[0]} R={parts[1]} TL={parts[2]} TR={parts[3]}"
        f"{' [关机]' if parts[4] == '1' else ''}"
    )
