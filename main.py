#!/usr/bin/env python3
"""水下机器人实验 — 统一控制台入口。

运行方式：
    source .venv/bin/activate
    python main.py

所有操作从下面的数字菜单进入，不需要记任何命令行参数。
真实推进器输出必须在进入模式时显式确认（默认模拟模式）。
"""

from __future__ import annotations

import sys

from rov.config import ROBOT_HOST, VIDEO_STREAM_URL, WEBSOCKET_URL

BANNER = """
╔══════════════════════════════════════════╗
║       水下机器人实验  控制台              ║
╚══════════════════════════════════════════╝"""

MENU = """
  1. 视频测试        看画面、测帧率（不发指令）
  2. 颜色标定        调 HSV 阈值（不发指令）
  3. 检测验证        看检测框效果（不发指令）
  4. 手动控制        WASD 驾驶 · 自动检测 · P 拍照
  5. 自主模式        自动找球→对准→碰触
  6. 系统检查        测试视频/通信连接
  7. 关闭机器人      发送关机指令
  0. 退出
──────────────────────────────────────────"""


def ask_live() -> bool:
    """问是否连接真实机器人（默认模拟）。"""
    print("\n⚠  将连接真实机器人并输出推进器指令。")
    print("   请确认：已留好物理急停方式，周围无障碍物。")
    answer = input("   连接真实机器人？[y/N]: ").strip().lower()
    return answer in ("y", "yes")


def mode_video() -> None:
    import video_test

    video_test.main(["--reconnect"])


def mode_color() -> None:
    import color_tune

    color_tune.main()


def mode_detect() -> None:
    from rov.modes import detect

    detect.run()


def mode_manual() -> None:
    from rov.modes import manual

    manual.run(live=ask_live())


def mode_auto() -> None:
    try:
        from rov.modes import auto
    except ImportError:
        print("\n自主模式尚未接入（下一块交付），先用「4. 手动控制」。")
        return
    auto.run(live=ask_live())


def mode_check() -> None:
    """系统检查：视频 + 通信连通性（短超时，不发推进器指令）。"""
    import threading

    import cv2

    print(f"\n机器人主机：{ROBOT_HOST}")
    print(f"视频流    ：{VIDEO_STREAM_URL}")
    print(f"WebSocket ：{WEBSOCKET_URL}\n")

    # 视频检查（后台线程带超时，防止 cv2 长阻塞）
    def check_video(results: list) -> None:
        cap = cv2.VideoCapture(VIDEO_STREAM_URL)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        results.append(bool(ok))

    video_result: list = []
    t = threading.Thread(target=check_video, args=(video_result,), daemon=True)
    t.start()
    t.join(timeout=8)
    if not video_result:
        print("❌ 视频检查超时（8 秒无响应）")
        print("   怎么办：确认机器人已开机、网线已连接、IP 是 192.168.137.100")
    elif video_result[0]:
        print("✅ 视频流正常")
    else:
        print("❌ 视频流打不开")
        print("   怎么办：用浏览器直接打开视频地址试试；确认机器人已开机")

    # 通信检查（只连不做动作，收到一帧传感器数据即成功）
    import asyncio

    async def check_ws() -> bool:
        import websockets

        try:
            async with websockets.connect(WEBSOCKET_URL, ping_interval=None) as ws:
                await ws.send("1462:1462:1462:1462:0")  # 中位停桨帧
                await asyncio.wait_for(ws.recv(), timeout=5)
                return True
        except Exception:  # noqa: BLE001
            return False

    try:
        ws_ok = asyncio.run(asyncio.wait_for(check_ws(), timeout=8))
    except asyncio.TimeoutError:
        ws_ok = False

    if ws_ok:
        print("✅ 通信正常（已收到传感器回传）")
    else:
        print("❌ 通信连接失败")
        print("   怎么办：确认机器人已就绪（开机有提示音）、端口 8000 未被占用")
        print(f"   也可以先跑本机模拟：python server.py（另开终端）")


def mode_shutdown() -> None:
    """发送关机指令（真实连接，需二次确认）。"""
    import asyncio

    from rov.comm import RobotComm

    print("\n⚠  将向真实机器人发送关机指令！")
    print("   后果：机器人将在几秒内关机，30 秒后需手动断电，本轮测试结束。")
    answer = input("   确认关机？输入 yes 才执行: ").strip().lower()
    if answer != "yes":
        print("已取消。")
        return

    async def do_shutdown() -> None:
        comm = RobotComm(WEBSOCKET_URL, live=True)
        await comm.start()
        await asyncio.sleep(1.0)  # 等连接建立
        comm.send_shutdown()
        await asyncio.sleep(1.0)  # 等指令发出
        await comm.stop()

    print("正在发送关机指令…")
    asyncio.run(do_shutdown())
    print("✅ 关机指令已发送。请等待约 30 秒后，按下电源按钮复位并断电。")


def main() -> int:
    actions = {
        "1": ("视频测试", mode_video),
        "2": ("颜色标定", mode_color),
        "3": ("检测验证", mode_detect),
        "4": ("手动控制", mode_manual),
        "5": ("自主模式", mode_auto),
        "6": ("系统检查", mode_check),
        "7": ("关闭机器人", mode_shutdown),
    }

    print(BANNER)
    print(f"机器人：{ROBOT_HOST}（默认模拟模式，真实输出需确认）")

    while True:
        print(MENU)
        try:
            choice = input("请选择 [0-7]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if choice in ("0", "q", "quit", "exit"):
            print("已退出。")
            return 0
        if choice not in actions:
            print("输入无效，请输入 0-7 的数字。")
            continue

        name, func = actions[choice]
        print(f"\n▶ 进入：{name}")
        try:
            func()
        except KeyboardInterrupt:
            print("\n已中断，返回主菜单。")
        except Exception as exc:  # noqa: BLE001 - 面向所有人的提示
            print(f"\n❌ {name} 出错：{exc}")
            print("   如果视频/连接问题，先运行「6. 系统检查」排查。")
        print(f"◀ 已退出：{name}")


if __name__ == "__main__":
    sys.exit(main())
