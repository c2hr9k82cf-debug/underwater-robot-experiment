"""rov.comm 单元测试（对本地自建 mock 服务器联调，不需要真实机器人）。"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import List

import pytest
import websockets

from rov.comm import COMMAND_STALE_S, RobotComm
from rov.thrusters import Attitude


class MockRobotServer:
    """协议兼容的本地模拟机器人（行为同 server.py：每条指令回 roll:pitch:yaw）。"""

    def __init__(self, reply: str = "0:1:180") -> None:
        self.received: List[str] = []
        self.reply = reply
        self.port: int = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None

    async def _handler(self, websocket, path) -> None:
        async for message in websocket:
            self.received.append(str(message))
            await websocket.send(self.reply)

    def start(self) -> None:
        ready = threading.Event()

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def serve() -> None:
                self._server = await websockets.serve(
                    self._handler, "127.0.0.1", 0, ping_interval=None
                )
                self.port = self._server.sockets[0].getsockname()[1]
                ready.set()

            self._loop.run_until_complete(serve())
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait(timeout=5)

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)


@pytest.fixture()
def mock_server() -> MockRobotServer:
    server = MockRobotServer()
    server.start()
    yield server
    server.stop()


def test_sim_mode_records_without_network() -> None:
    """模拟模式：不连接任何服务器，但 last_command 正确记录。"""
    async def scenario() -> None:
        comm = RobotComm("ws://127.0.0.1:1", live=False)  # 不存在的地址也不应连接
        await comm.start()
        comm.set_thrust(0.305, 0.0, 0.0)
        await asyncio.sleep(0.3)
        assert comm.last_command == "1523:1401:1462:1462:0"
        assert comm.connected is False
        await comm.stop()

    asyncio.run(scenario())


def test_stale_target_returns_neutral() -> None:
    """目标推力超时未更新 → 自动回中位（防失控）。"""
    async def scenario() -> None:
        comm = RobotComm("ws://127.0.0.1:1", live=False)
        await comm.start()
        comm.set_thrust(1.0, 0.0, 0.0)
        await asyncio.sleep(0.2)
        assert comm.last_command != "1462:1462:1462:1462:0"
        await asyncio.sleep(COMMAND_STALE_S + 0.3)
        assert comm.last_command == "1462:1462:1462:1462:0"
        await comm.stop()

    asyncio.run(scenario())


def test_live_send_and_receive(mock_server: MockRobotServer) -> None:
    """真实模式：指令送达 mock 服务器，传感器回传被解析。"""
    async def scenario() -> None:
        comm = RobotComm(f"ws://127.0.0.1:{mock_server.port}", live=True)
        await comm.start()
        comm.set_thrust(0.305, 0.0, 0.0)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not mock_server.received:
            await asyncio.sleep(0.05)

        assert mock_server.received, "mock 服务器未收到任何指令"
        assert mock_server.received[0] == "1523:1401:1462:1462:0"

        await asyncio.sleep(0.3)
        assert comm.latest_attitude == Attitude(roll=0.0, pitch=1.0, yaw=180.0)
        await comm.stop()

    asyncio.run(scenario())


def test_heartbeat_continues(mock_server: MockRobotServer) -> None:
    """心跳：不更新目标也持续发指令（防推进器鸣叫）。"""
    async def scenario() -> None:
        comm = RobotComm(f"ws://127.0.0.1:{mock_server.port}", live=True)
        await comm.start()
        await asyncio.sleep(0.6)
        assert len(mock_server.received) >= 3, f"心跳不足：{len(mock_server.received)} 帧"
        await comm.stop()

    asyncio.run(scenario())


def test_stop_sends_neutral_frames(mock_server: MockRobotServer) -> None:
    """退出时补发中位停桨帧。"""
    async def scenario() -> None:
        comm = RobotComm(f"ws://127.0.0.1:{mock_server.port}", live=True)
        await comm.start()
        comm.set_thrust(1.0, 1.0, 1.0)
        await asyncio.sleep(0.3)
        await comm.stop()

        assert mock_server.received, "停桨帧未送达"
        assert mock_server.received[-1] == "1462:1462:1462:1462:0"
        assert comm.connected is False

    asyncio.run(scenario())


def test_shutdown_command_sent(mock_server: MockRobotServer) -> None:
    """关机指令显式发送（flag=1）。"""
    async def scenario() -> None:
        comm = RobotComm(f"ws://127.0.0.1:{mock_server.port}", live=True)
        await comm.start()
        await asyncio.sleep(0.3)
        comm.send_shutdown()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if any(cmd.endswith(":1") for cmd in mock_server.received):
                break
            await asyncio.sleep(0.05)
        assert any(cmd.endswith(":1") for cmd in mock_server.received), "未收到关机指令"
        await comm.stop()

    asyncio.run(scenario())
