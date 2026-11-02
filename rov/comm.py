"""WebSocket 通信客户端模块。

职责：
- 连接机器人 WebSocket（机器人是服务端，本机是客户端）
- 定频发送循环（默认 10 Hz）持续发推进器指令——推进器长时间不收
  指令会鸣叫，所以哪怕停桨也要持续发中位帧
- 目标推力超过 0.5 秒未更新自动回中位（防失控）
- 接收循环解析传感器回传（roll:pitch:yaw）
- 断线自动重连；退出/异常时发中位停桨帧再断开

两种连接模式：
- 真实模式（live）：连接 WEBSOCKET_URL，真实发送指令
- 模拟模式（sim）：不连接任何服务器，指令记录在 last_command 供界面显示

关机指令（flag=1）只能通过 send_shutdown() 显式发送，不做自动重试。
本模块不包含任何决策逻辑（那是 rov/autonomy.py 的职责）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from rov.thrusters import (
    Attitude,
    build_command,
    mix,
    parse_sensor,
    stop_command,
)

logger = logging.getLogger(__name__)

# 发送频率（Hz）。推进器鸣叫看门狗大约秒级，10 Hz 足够。
SEND_RATE_HZ: float = 10.0
# 目标推力超过该时间未更新则自动回中位（秒）
COMMAND_STALE_S: float = 0.5
# 断线重连间隔（秒）
RECONNECT_DELAY_S: float = 2.0
# 退出时补发中位帧次数
STOP_FRAMES_ON_EXIT: int = 3


class RobotComm:
    """机器人通信客户端（真实/模拟两用）。"""

    def __init__(self, url: str, *, live: bool = False) -> None:
        self._url = url
        self._live = live

        self._ws = None                      # websockets 连接（live 模式）
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._running = False

        # 最新目标推力（surge, yaw, heave）与更新时间
        self._target = (0.0, 0.0, 0.0)
        self._target_time = 0.0
        # 最近一条实际发出的指令（模拟模式下界面显示用）
        self.last_command: str = stop_command()
        # 最近一次传感器数据
        self.latest_attitude: Optional[Attitude] = None
        self.connected = False

    # ------------------------------------------------------------------ #
    # 连接生命周期
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """启动发送/接收循环（模拟模式只记录不发送）。"""
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._send_task = asyncio.create_task(self._send_loop())
        if self._live:
            self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("通信已启动：%s", "真实连接 " + self._url if self._live else "模拟模式")

    async def stop(self) -> None:
        """停止并补发中位停桨帧，安全收尾。"""
        self._running = False
        for task in (self._send_task, self._recv_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._send_task = self._recv_task = None

        if self._live and self._ws is not None:
            try:
                for _ in range(STOP_FRAMES_ON_EXIT):
                    await self._ws.send(stop_command())
            except Exception:  # noqa: BLE001 - 收尾失败不再追究
                pass
            await self._ws.close()
            self._ws = None
            self.connected = False
        logger.info("通信已停止（已发 %d 个中位停桨帧）", STOP_FRAMES_ON_EXIT)

    # ------------------------------------------------------------------ #
    # 指令输入
    # ------------------------------------------------------------------ #

    def set_thrust(self, surge: float, yaw: float, heave: float) -> None:
        """更新目标推力（[-1, 1]）。界面/决策层每帧调用。"""
        self._target = (surge, yaw, heave)
        self._target_time = time.monotonic()

    def send_shutdown(self) -> None:
        """发送关机指令（flag=1）。仅限用户显式请求，不重试。

        线程安全：UI 线程可直接调用，内部调度到事件循环执行。
        """
        cmd = build_command(mix(0.0, 0.0, 0.0), shutdown=True)
        self.last_command = cmd
        if self._live and self._ws is not None and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self._ws.send(cmd), self._loop)
                future.result(timeout=2)
                logger.warning("已发送关机指令：%s", cmd)
            except Exception as exc:  # noqa: BLE001
                logger.error("关机指令发送失败：%s", exc)
        else:
            logger.warning("模拟模式，关机指令未真实发送：%s", cmd)

    # ------------------------------------------------------------------ #
    # 内部循环
    # ------------------------------------------------------------------ #

    async def _send_loop(self) -> None:
        """定频发送指令；目标过期自动回中位。模拟模式只记录不发送。"""
        interval = 1.0 / SEND_RATE_HZ
        while self._running:
            try:
                if self._live and self._ws is None:
                    await self._connect()
                    continue

                if time.monotonic() - self._target_time > COMMAND_STALE_S:
                    surge, yaw, heave = 0.0, 0.0, 0.0
                else:
                    surge, yaw, heave = self._target

                cmd = build_command(mix(surge, yaw, heave))
                self.last_command = cmd
                if self._live and self._ws is not None:
                    await self._ws.send(cmd)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 断线交给下轮重连
                self.connected = False
                self._ws = None
                logger.warning("发送失败，%.1f 秒后重连", RECONNECT_DELAY_S)
            await asyncio.sleep(interval)

    async def _recv_loop(self) -> None:
        """接收传感器回传。"""
        while self._running:
            try:
                if self._ws is None:
                    await asyncio.sleep(0.2)
                    continue
                message = await self._ws.recv()
                if isinstance(message, bytes):
                    message = message.decode(errors="replace")
                attitude = parse_sensor(message)
                if attitude is not None:
                    self.latest_attitude = attitude
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self.connected = False
                self._ws = None

    async def _connect(self) -> None:
        """建立连接（带重试）。"""
        try:
            import websockets  # 延迟导入，模拟模式不需要

            self._ws = await websockets.connect(self._url, ping_interval=None)
            self.connected = True
            logger.info("已连接机器人：%s", self._url)
        except Exception:  # noqa: BLE001
            self.connected = False
            self._ws = None
            await asyncio.sleep(RECONNECT_DELAY_S)
