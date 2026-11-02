"""MJPEG 视频流读取模块。

职责：
- 打开/关闭 HTTP MJPEG 视频流
- 逐帧读取并计算 FPS
- 检测断流（超时无新帧）
- 可选自动重连

本模块不包含任何电机/推进器控制逻辑。
"""

from __future__ import annotations

import logging
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
    """对 cv2.VideoCapture 的薄封装，带 FPS 统计与断流检测。"""

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
        self._last_frame_time: float = 0.0
        self._fps_timestamps: Deque[float] = deque(maxlen=FPS_WINDOW)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def open(self) -> None:
        """打开视频流。失败时抛出 VideoStreamError。"""
        if self._cap is not None:
            self.close()

        logger.info("正在连接视频流：%s", self._url)
        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            cap.release()
            raise VideoStreamError(f"无法打开视频流：{self._url}")

        self._cap = cap
        self._last_frame_time = time.monotonic()
        self._fps_timestamps.clear()
        logger.info("视频流已连接")

    def close(self) -> None:
        """释放底层资源。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
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
        """读取一帧。

        成功返回 FramePacket；断流且未开启重连时返回 None；
        开启重连时内部循环重试，直到成功或调用方关闭。
        """
        while True:
            if not self.is_open:
                if not self._reconnect:
                    return None
                self._try_reconnect()

            assert self._cap is not None
            ok, frame = self._cap.read()
            now = time.monotonic()

            if not ok or frame is None:
                if now - self._last_frame_time > self._frame_timeout_s:
                    logger.warning("视频流超时（%.1f 秒无新帧）", self._frame_timeout_s)
                    self.close()
                    if not self._reconnect:
                        return None
                    continue
                # 偶发单帧失败，短暂等待后重试
                time.sleep(0.01)
                continue

            self._last_frame_time = now
            self._fps_timestamps.append(now)
            return FramePacket(frame=frame, timestamp=now, fps=self._current_fps())

    def _try_reconnect(self) -> None:
        """按固定间隔尝试重连，直到成功。"""
        while self._cap is None or not self._cap.isOpened():
            try:
                self.open()
            except VideoStreamError:
                logger.info("%.1f 秒后重试连接…", self._reconnect_delay_s)
                time.sleep(self._reconnect_delay_s)

    def _current_fps(self) -> float:
        if len(self._fps_timestamps) < 2:
            return 0.0
        elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._fps_timestamps) - 1) / elapsed
