"""MJPEG 视频流读取模块。

职责：
- 打开/关闭 HTTP MJPEG 视频流
- 后台线程逐帧解码，主线程非阻塞取最新帧（cv2 网络流 read() 会长时间
  阻塞，若在主循环里直接调，视频一抖动主循环就卡死 → 推力停止更新 →
  通信层自动回中位 → 机器人"动一下就不动了"）
- 检测断流（超时无新帧）
- 可选自动重连

本模块不包含任何电机/推进器控制逻辑。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import cv2
import numpy as np

from rov.config import FPS_WINDOW, FRAME_TIMEOUT_S, RECONNECT_DELAY_S

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FramePacket:
    """一帧图像及其元信息。"""

    frame: np.ndarray
    timestamp: float
    fps: float


class VideoStreamError(RuntimeError):
    """视频流打开或读取失败。"""


class VideoStream:
    """对 cv2.VideoCapture 的封装：后台线程解码，read() 非阻塞。"""

    def __init__(
        self,
        url: str,
        *,
        frame_timeout_s: float = FRAME_TIMEOUT_S,
        reconnect: bool = False,
        reconnect_delay_s: float = RECONNECT_DELAY_S,
    ) -> None:
        self._url = url
        self._frame_timeout_s = frame_timeout_s
        self._reconnect = reconnect
        self._reconnect_delay_s = reconnect_delay_s

        self._cap: Optional[cv2.VideoCapture] = None
        self._fps_timestamps: Deque[float] = deque(maxlen=FPS_WINDOW)
        self._latest: Optional[FramePacket] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def open(self) -> None:
        """打开视频流并启动后台读帧线程。失败时抛出 VideoStreamError。"""
        if self._cap is not None:
            self.close()

        logger.info("正在连接视频流：%s", self._url)
        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            cap.release()
            raise VideoStreamError(f"无法打开视频流：{self._url}")

        self._stop.clear()
        self._cap = cap
        self._fps_timestamps.clear()
        self._thread = threading.Thread(
            target=self._reader_loop, name="video-reader", daemon=True
        )
        self._thread.start()
        logger.info("视频流已连接")

    def close(self) -> None:
        """释放底层资源并停止读帧线程。"""
        self._stop.set()
        if self._cap is not None:
            self._cap.release()   # 通常能打断阻塞中的 read()
            self._cap = None
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)   # 卡死的线程是 daemon，不阻塞退出
        with self._lock:
            self._latest = None
        logger.info("视频流已关闭")

    def __enter__(self) -> "VideoStream":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Optional[FramePacket]:
        """非阻塞取最新帧。

        后台线程持续解码，本方法只看一眼最新帧的时间戳：
        - 新鲜 → 返回该帧（可能与上一帧相同，无害）
        - 超时无新帧/线程已退出 → 视为断流，close 后返回 None；
          reconnect=True 时自动尝试重开
        """
        with self._lock:
            packet = self._latest
        thread_alive = self._thread is not None and self._thread.is_alive()
        if (
            packet is not None
            and thread_alive
            and time.monotonic() - packet.timestamp <= self._frame_timeout_s
        ):
            return packet

        if self._cap is not None:
            logger.warning("视频流断流（%.1f 秒无新帧）", self._frame_timeout_s)
            self.close()
            if self._reconnect:
                try:
                    self.open()
                except VideoStreamError:
                    logger.info("%.1f 秒后重试连接…", self._reconnect_delay_s)
        return None

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _reader_loop(self) -> None:
        """后台读帧线程。cv2 网络流 read() 可能长期阻塞，隔离在这里。"""
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                return
            try:
                ok, frame = cap.read()
            except Exception:  # noqa: BLE001 - cap 被释放等，线程直接退出
                return
            if self._stop.is_set():
                return
            now = time.monotonic()
            if not ok or frame is None:
                # 偶发单帧失败；连续断流由 read() 侧按时间戳判定
                time.sleep(0.01)
                continue
            self._fps_timestamps.append(now)
            packet = FramePacket(frame=frame, timestamp=now, fps=self._current_fps())
            with self._lock:
                self._latest = packet

    def _current_fps(self) -> float:
        if len(self._fps_timestamps) < 2:
            return 0.0
        elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._fps_timestamps) - 1) / elapsed
