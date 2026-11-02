"""手动控制模式（WASD 驾驶 + 自动检测 + 一键保存图像）。

按键设计为"按住生效、松开自动回停"（约 0.5 秒无按键即回中位），
这是刻意的安全设计：手一松机器人就停。空格立即停桨。

  W / S      前进 / 后退
  A / D      左转 / 右转
  R / F      上浮 / 下潜
  1 / 2 / 3  速度档（慢 0.35 / 中 0.6 / 快 1.0）
  P          保存当前图像（存 snapshots/，原图便于标定）
  空格       立即停桨
  T          开关检测框叠加（默认开）
  Q / ESC    退出（自动发停桨帧）

终端只在关键事件打一行，状态全部画在视频窗口左上角。
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import cv2

from rov.config import VIDEO_STREAM_URL, WINDOW_NAME
from rov.detector import BallDetector
from rov.modes.runtime import (
    CommRunner,
    draw_hud,
    placeholder_frame,
    pwm_hud_line,
    save_snapshot,
)
from rov.video_stream import VideoStream, VideoStreamError

# 按键 → (轴, 方向)。yaw 正=右转，负=左转
KEY_MAP: Dict[int, Tuple[str, int]] = {
    ord("w"): ("surge", 1),
    ord("s"): ("surge", -1),
    ord("a"): ("yaw", -1),
    ord("d"): ("yaw", 1),
    ord("r"): ("heave", 1),
    ord("f"): ("heave", -1),
}

SPEED_LEVELS = [0.35, 0.6, 1.0]
SPEED_LABELS = ["慢", "中", "快"]
# 按键保持时间（秒）：超过该时间无按键，该轴自动回 0
HOLD_S = 0.5
REOPEN_DELAY_S = 2.0


def run(live: bool = False, video_url: str = VIDEO_STREAM_URL) -> None:
    """手动控制主循环。live=True 需调用方已完成确认。"""
    from rov.config import WEBSOCKET_URL

    comm = CommRunner(WEBSOCKET_URL, live=live)
    detector = BallDetector()
    show_detect = True  # 自动检测默认开
    speed_idx = 0
    axes: Dict[str, Tuple[float, float]] = {"surge": (0.0, 0.0), "yaw": (0.0, 0.0), "heave": (0.0, 0.0)}
    saved_note: Optional[str] = None
    saved_at = 0.0

    stream = VideoStream(video_url, reconnect=False, frame_timeout_s=1.5)
    window = f"{WINDOW_NAME} - 手动控制"
    last_open_attempt = 0.0
    fps = 0.0

    print("─── 手动控制 ─────────────────────────────")
    print("  W/S 前进后退   A/D 左转右转   R/F 上浮下潜")
    print("  1/2/3 速度档   P 保存图像    空格 立即停桨")
    print("  T 检测框开关（默认开）   Q / ESC 退出")
    print("  （松开按键约 0.5 秒自动回停）")
    print(f"  连接模式：{'真实机器人' if live else '模拟（不发指令）'}")
    print("──────────────────────────────────────────")

    comm.start()
    try:
        try:
            stream.open()
            print(f"视频已连接：{video_url}")
        except VideoStreamError:
            print("视频暂时打不开，将持续重试（仍可退出）")

        while True:
            now = time.monotonic()

            # 视频（断了就重试，不阻塞按键）
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

            # 按键（每帧最多一个；按住靠系统自动重复）
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q / ESC
                break
            if key == ord(" "):
                axes = {k: (0.0, 0.0) for k in axes}
                print("已停桨")
            elif key == ord("p"):
                # 保存当前帧原图（标注前的画面，便于后续标定）
                if frame is None:
                    saved_note = "视频断开，无法保存"
                else:
                    path = save_snapshot(frame.copy())
                    saved_note = f"已保存 {path.name}"
                    print(f"已保存 {path}")
                saved_at = now
            elif key == ord("t"):
                show_detect = not show_detect
                print(f"检测框：{'开' if show_detect else '关'}")
            elif key in (ord("1"), ord("2"), ord("3")):
                speed_idx = key - ord("1")
            elif key in KEY_MAP:
                axis, sign = KEY_MAP[key]
                axes[axis] = (sign * SPEED_LEVELS[speed_idx], now)

            # 各轴过期自动回 0
            values = {}
            for axis, (val, ts) in axes.items():
                values[axis] = val if now - ts < HOLD_S else 0.0

            comm.set_thrust(values["surge"], values["yaw"], values["heave"])

            if frame is None:
                frame = placeholder_frame()
            else:
                if show_detect:
                    result = detector.detect(frame)
                    frame = detector.annotate(frame, result)

            # HUD
            conn = "已连接" if comm.connected else ("未连接" if live else "模拟")
            lines = [
                f"手动控制 | 速度档 {speed_idx + 1}({SPEED_LABELS[speed_idx]}) | {conn}",
                f"推力 前后 {values['surge']:+.2f}  转向 {values['yaw']:+.2f}  升沉 {values['heave']:+.2f}",
                pwm_hud_line(comm.last_command),
            ]
            att = comm.attitude
            if att is not None:
                lines.append(f"姿态 roll {att.roll:.1f} pitch {att.pitch:.1f} yaw {att.yaw:.1f}")
            else:
                lines.append("姿态 等待数据...")
            if show_detect:
                lines.append("检测框 T=开 | P 保存图像")
            else:
                lines.append("检测框 T=关 | P 保存图像")
            if saved_note is not None and now - saved_at < 2.5:
                lines.append(saved_note)
            lines.append(f"FPS {fps:.1f} | Q/ESC 退出 空格停桨")
            draw_hud(frame, lines)
            cv2.imshow(window, frame)
    finally:
        comm.stop()
        stream.close()
        cv2.destroyWindow(window)
        print("手动控制已退出（已发停桨帧）")
