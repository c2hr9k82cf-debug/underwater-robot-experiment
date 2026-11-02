"""检测验证模式（菜单 3）：只看检测框，不发任何指令。

  T  开关原始掩膜叠加
  P  保存当前图像（存 snapshots/，原图便于标定）
  Q / ESC  返回主菜单
"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from rov.config import VIDEO_STREAM_URL, WINDOW_NAME
from rov.detector import BallDetector
from rov.modes.runtime import draw_hud, placeholder_frame, save_snapshot
from rov.video_stream import VideoStream, VideoStreamError

REOPEN_DELAY_S = 2.0


def run(video_url: str = VIDEO_STREAM_URL) -> None:
    detector = BallDetector()
    stream = VideoStream(video_url, reconnect=False, frame_timeout_s=1.5)
    window = f"{WINDOW_NAME} - 检测验证"
    show_mask = False
    saved_note: Optional[str] = None
    saved_at = 0.0
    last_open_attempt = 0.0
    fps = 0.0

    print("─── 检测验证 ─────────────────────────────")
    print("  只看画面和检测框，不会发送任何推进器指令")
    print("  T 开关掩膜叠加   P 保存图像   Q / ESC 返回主菜单")
    print("──────────────────────────────────────────")

    try:
        try:
            stream.open()
            print(f"视频已连接：{video_url}")
        except VideoStreamError:
            print("视频暂时打不开，将持续重试")

        while True:
            now = time.monotonic()
            frame = None
            if stream.is_open:
                packet = stream.read()
                if packet is not None:
                    frame = packet.frame
                    fps = packet.fps
            else:
                if now - last_open_attempt > REOPEN_DELAY_S:
                    last_open_attempt = now
                    try:
                        stream.open()
                        print("视频已重连")
                    except VideoStreamError:
                        pass

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("t"):
                show_mask = not show_mask
            elif key == ord("p"):
                # 保存当前帧原图（标注前的画面，便于后续标定）
                if frame is None:
                    saved_note = "视频断开，无法保存"
                else:
                    path = save_snapshot(frame.copy())
                    saved_note = f"已保存 {path.name}"
                    print(f"已保存 {path}")
                saved_at = now

            if frame is None:
                frame = placeholder_frame()
                status = "视频断开"
            else:
                result = detector.detect(frame)
                frame = detector.annotate(frame, result)
                if show_mask:
                    mask = detector.build_mask(frame)
                    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                status = "LOST 未检测到" if not result.is_detected else (
                    f"DETECTED 中心({result.center[0]:.0f},{result.center[1]:.0f})"
                )

            hud_lines = [
                "检测验证（无指令输出）",
                status,
            ]
            if saved_note is not None and now - saved_at < 2.5:
                hud_lines.append(saved_note)
            hud_lines.append(f"FPS {fps:.1f} | T 掩膜 P 保存 | Q/ESC 返回")
            draw_hud(frame, hud_lines)
            cv2.imshow(window, frame)
    finally:
        stream.close()
        cv2.destroyWindow(window)
        print("检测验证已退出")
