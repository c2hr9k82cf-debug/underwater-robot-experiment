#!/usr/bin/env python3
"""阶段 1：视频采集测试程序。

从机器人 MJPEG 视频流读取实时画面并显示，叠加 FPS 信息。
按 q 退出。

本程序 **不会** 发送任何电机/推进器控制指令。
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from rov.config import RECONNECT_DELAY_S, VIDEO_STREAM_URL, WINDOW_NAME
from rov.video_stream import VideoStream, VideoStreamError

logger = logging.getLogger("video_test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROV 摄像头视频采集测试（无电机控制）")
    parser.add_argument(
        "--url",
        default=VIDEO_STREAM_URL,
        help=f"MJPEG 视频流地址（默认：{VIDEO_STREAM_URL}）",
    )
    parser.add_argument(
        "--reconnect",
        action="store_true",
        help="断流后自动重连（默认直接退出）",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=RECONNECT_DELAY_S,
        help=f"重连间隔秒数（默认：{RECONNECT_DELAY_S}）",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="输出调试日志",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """主显示循环。返回进程退出码。"""
    try:
        stream = VideoStream(
            args.url,
            reconnect=args.reconnect,
            reconnect_delay_s=args.reconnect_delay,
        )
        stream.open()
    except VideoStreamError as exc:
        logger.error("%s", exc)
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print("视频采集已启动。按 q 退出。")
    exit_code = 0

    try:
        with stream:
            while True:
                packet = stream.read()
                if packet is None:
                    logger.error("视频流中断，程序退出")
                    print("错误：视频流中断，程序退出。", file=sys.stderr)
                    exit_code = 1
                    break

                frame = packet.frame
                info = f"FPS: {packet.fps:.1f}"
                cv2.putText(
                    frame,
                    info,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("用户按 q 退出")
                    break
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在退出…")
        logger.info("收到 KeyboardInterrupt，退出")
    finally:
        cv2.destroyAllWindows()

    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
