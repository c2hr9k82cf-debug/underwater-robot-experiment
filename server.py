import asyncio
import websockets

async def handler(websocket, path):
    print("客户端已连接")

    async for message in websocket:
        print("收到控制指令：", message)

        parts = message.split(":")
        if len(parts) == 5:
            print("指令格式正确")
            if parts[-1] == "0":
                print("普通控制模式")
            elif parts[-1] == "1":
                print("收到关机标志")
        else:
            print("指令格式错误")

        await websocket.send("10:20:30")
        print("已返回模拟传感器数据：10:20:30")

async def main():
    async with websockets.serve(
        handler, "127.0.0.1", 8000, ping_interval=None
    ):
        print("模拟服务器已启动：ws://127.0.0.1:8000")
        await asyncio.Future()

asyncio.run(main())
