"""自主模式（菜单 5）：自动找球 → 对准 → 接近 → 轻触 → 后退。

决策来自 rov.autonomy 状态机（纯逻辑，已单测），
本模块只负责：取帧 → 检测 → 喂状态机 → 发推力 → HUD 显示。

  空格       暂停/恢复（暂停时发停桨帧）
  R          重新开始任务
  T          检测框开关
  Q / ESC    退出（自动发停桨帧）

终端只在状态切换时打一行，状态大字显示在画面上。
"""

from __future__ import annotations

import time

import cv2

from rov.config import VIDEO_STREAM_URL, WEBSOCKET_URL, WINDOW_NAME
from rov.autonomy import AutonomyContext, AutonomyState, step
from rov.detector import BallDetector
from rov.modes.runtime import CommRunner, draw_hud, placeholder_frame, pwm_hud_line
from rov.video_stream import VideoStream, VideoStreamError

REOPEN_DELAY_S = 2.0

STATE_LABELS = {
    AutonomyState.SEARCH: "搜索中 SEARCH",
    AutonomyState.ALIGN: "对准中 ALIGN",
    AutonomyState.APPROACH: "接近中 APPROACH",
    AutonomyState.TOUCH: "轻触中 TOUCH",
    AutonomyState.BACKOFF: "后退中 BACKOFF",
    AutonomyState.DONE: "完成 DONE",
}


def run(live: bool = False, video_url: str = VIDEO_STREAM_URL) -> None:
    """自主模式主循环。live=True 需调用方已完成确认。"""
    comm = CommRunner(WEBSOCKET_URL, live=live)
    detector = BallDetector()
    ctx = AutonomyContext()
    paused = False
    show_detect = False
    last_state = None
    stream = VideoStream(video_url, reconnect=False, frame_timeout_s=1.5)
    window = f"{WINDOW_NAME} - 自主模式"
    last_open_attempt = 0.0
    fps = 0.0

    print("─── 自主模式 ─────────────────────────────")
    print("  机器人自动执行：搜索 → 对准 → 接近 → 轻触 → 后退")
    print("  空格 暂停/恢复    R 重新开始    T 检测框")
    print("  Q / ESC 退出（自动发停桨帧）")
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
            if key == ord(" "):
                paused = not paused
                print(f"{'已暂停' if paused else '已恢复'}")
            elif key == ord("r"):
                ctx = AutonomyContext()
                paused = False
                print("任务已重新开始")
            elif key == ord("t"):
                show_detect = not show_detect

            # 决策 + 发指令
            if frame is not None and not paused and not ctx.done:
                result = detector.detect(frame)
                out = step(ctx, result, (frame.shape[1], frame.shape[0]), now)
                comm.set_thrust(out.surge, out.yaw, out.heave)
                if out.state is not last_state:
                    if out.note:
                        print(f"[{out.state.value}] {out.note}")
                    last_state = out.state
                if show_detect:
                    frame = detector.annotate(frame, result)
                state_label = STATE_LABELS[out.state]
                state_note = out.note
            elif paused:
                comm.set_thrust(0.0, 0.0, 0.0)
                state_label = "已暂停 PAUSED"
                state_note = "按空格恢复"
            else:
                state_label = STATE_LABELS[ctx.state]
                state_note = "任务已完成，按 R 重新开始" if ctx.done else "等待视频"

            if frame is None:
                frame = placeholder_frame()

            conn = "已连接" if comm.connected else ("未连接" if live else "模拟")
            draw_hud(frame, [
                f"自主模式 | {state_label} | {conn}",
                state_note or "",
                pwm_hud_line(comm.last_command),
                f"FPS {fps:.1f} | 空格暂停 R重来 Q/ESC退出",
            ])
            # 状态大字画在画面下方
            cv2.putText(
                frame, state_label, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow(window, frame)
    finally:
        comm.stop()
        stream.close()
        cv2.destroyWindow(window)
        print("自主模式已退出（已发停桨帧）")
