#!/usr/bin/env python3
"""阶段 2：HSV 阈值调试工具。

用滑块实时调整 HSV 范围，预览掩膜与检测效果。
支持三种输入源：
  1. 实时视频流（默认，需连接机器人）
  2. 本地图片文件（--image）
  3. 本地视频文件（--video）

按键：
  q / ESC  退出
  s        打印当前 HSV 阈值到控制台
  g / r    切换目标颜色（green / red）

本程序 **不会** 发送任何电机/推进器控制指令。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from rov.config import HSV_RANGES, TARGET_COLOR, VIDEO_STREAM_URL, TargetColor
from rov.detector import BallDetector
from rov.video_stream import VideoStream, VideoStreamError

logger = logging.getLogger("color_debug")

WINDOW_ORIGINAL = "Original"
WINDOW_MASK = "Mask"
WINDOW_TRACKBARS = "HSV Trackbars"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSV 阈值调试工具（无电机控制）")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="使用本地图片作为输入")
    source.add_argument("--video", type=Path, help="使用本地视频文件作为输入")
    parser.add_argument(
        "--url",
        default=VIDEO_STREAM_URL,
        help=f"实时视频流地址（默认：{VIDEO_STREAM_URL}）",
    )
    parser.add_argument(
        "--color",
        choices=["green", "red"],
        default=TARGET_COLOR,
        help=f"目标颜色（默认：{TARGET_COLOR}）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    return parser.parse_args(argv)


# ------------------------------------------------------------------ #
# Trackbar UI
# ------------------------------------------------------------------ #


class HsvTuner:
    """HSV 滑块调节器。"""

    def __init__(self, color: TargetColor) -> None:
        self.color: TargetColor = color
        # 当前阈值（lower / upper）
        seg = HSV_RANGES[color]
        self.lower: np.ndarray = np.array(seg["lower"], dtype=np.int32)
        self.upper: np.ndarray = np.array(seg["upper"], dtype=np.int32)

        cv2.namedWindow(WINDOW_TRACKBARS, cv2.WINDOW_NORMAL)
        self._create_trackbar("H Low", 0, 179, self.lower[0])
        self._create_trackbar("S Low", 0, 255, self.lower[1])
        self._create_trackbar("V Low", 0, 255, self.lower[2])
        self._create_trackbar("H High", 0, 179, self.upper[0])
        self._create_trackbar("S High", 0, 255, self.upper[1])
        self._create_trackbar("V High", 0, 255, self.upper[2])

    @staticmethod
    def _create_trackbar(name: str, max_val: int, _unused: int, init: int) -> None:
        # 避免命名冲突，所有 trackbar 挂在同一个窗口
        cv2.createTrackbar(name, WINDOW_TRACKBARS, int(init), max_val, lambda _v: None)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """读取当前滑块值，返回 (lower, upper)。"""
        names = ["H Low", "S Low", "V Low", "H High", "S High", "V High"]
        values = [cv2.getTrackbarPos(n, WINDOW_TRACKBARS) for n in names]
        self.lower = np.array(values[:3], dtype=np.int32)
        self.upper = np.array(values[3:], dtype=np.int32)
        return self.lower, self.upper

    def set_color(self, color: TargetColor) -> None:
        """切换目标颜色并重置滑块。"""
        self.color = color
        seg = HSV_RANGES[color]
        for name, val in zip(["H Low", "S Low", "V Low"], seg["lower"]):
            cv2.setTrackbarPos(name, WINDOW_TRACKBARS, val)
        for name, val in zip(["H High", "S High", "V High"], seg["upper"]):
            cv2.setTrackbarPos(name, WINDOW_TRACKBARS, val)
        logger.info("目标颜色切换为：%s", color)

    def print_values(self) -> None:
        lo = self.lower
        hi = self.upper
        print(f"\n当前 HSV（color={self.color}）：")
        print(f"  lower: ({lo[0]}, {lo[1]}, {lo[2]})")
        print(f"  upper: ({hi[0]}, {hi[1]}, {hi[2]})")
        print(f"  可直接写入 rov/config.py HSV_RANGES['{self.color}']\n")


# ------------------------------------------------------------------ #
# 输入源
# ------------------------------------------------------------------ #


class FrameSource:
    """统一的帧读取接口。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._stream: VideoStream | None = None
        self._video_cap: cv2.VideoCapture | None = None
        self._static_frame: np.ndarray | None = None

    def open(self) -> None:
        if self._args.image is not None:
            path = str(self._args.image)
            frame = cv2.imread(path)
            if frame is None:
                raise FileNotFoundError(f"无法读取图片：{path}")
            self._static_frame = frame
            logger.info("使用静态图片：%s", path)

        elif self._args.video is not None:
            path = str(self._args.video)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise FileNotFoundError(f"无法打开视频：{path}")
            self._video_cap = cap
            logger.info("使用本地视频：%s", path)

        else:
            self._stream = VideoStream(self._args.url)
            self._stream.open()
            logger.info("使用实时视频流：%s", self._args.url)

    def read(self) -> np.ndarray | None:
        if self._static_frame is not None:
            return self._static_frame.copy()
        if self._video_cap is not None:
            ok, frame = self._video_cap.read()
            return frame if ok else None
        if self._stream is not None:
            packet = self._stream.read()
            return packet.frame if packet else None
        return None

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        if self._video_cap is not None:
            self._video_cap.release()


# ------------------------------------------------------------------ #
# 主循环
# ------------------------------------------------------------------ #


def run(args: argparse.Namespace) -> int:
    source = FrameSource(args)
    try:
        source.open()
    except (VideoStreamError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    tuner = HsvTuner(args.color)
    color: TargetColor = args.color

    print("HSV 调试工具已启动。")
    print("  拖动滑块调整阈值")
    print("  q/ESC 退出 | s 打印阈值 | g 绿球 | r 红球")

    try:
        while True:
            frame = source.read()
            if frame is None:
                if args.image is not None:
                    frame = source.read()  # 静态图不会 None
                    if frame is None:
                        break
                else:
                    print("视频流结束或中断。", file=sys.stderr)
                    break

            lower, upper = tuner.read()

            # 用当前阈值构建检测器（红色会自动合并两段）
            if color == "red":
                detector = BallDetector(
                    color="red",
                    hsv_lower=tuple(int(v) for v in lower),
                    hsv_upper=tuple(int(v) for v in upper),
                )
                # 重新绑定单段（调试时滑块只控制一段）
                detector._hsv_segments = [
                    ((int(lower[0]), int(lower[1]), int(lower[2])),
                     (int(upper[0]), int(upper[1]), int(upper[2])))
                ]
            else:
                detector = BallDetector(
                    color=color,
                    hsv_lower=tuple(int(v) for v in lower),
                    hsv_upper=tuple(int(v) for v in upper),
                )

            result, candidates = detector.detect_with_candidates(frame)
            mask = detector.build_mask(frame)
            annotated = detector.annotate(frame.copy(), result, draw_candidates=candidates)

            cv2.imshow(WINDOW_ORIGINAL, annotated)
            cv2.imshow(WINDOW_MASK, mask)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                tuner.print_values()
            elif key == ord("g"):
                color = "green"
                tuner.set_color("green")
            elif key == ord("r"):
                color = "red"
                tuner.set_color("red")

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，退出。")
    finally:
        source.close()
        cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
