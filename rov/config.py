"""项目共享配置常量。

所有地址和可调参数集中在此，避免散落在各模块中。
真实硬件测试前请按现场情况确认这些值。
"""

from __future__ import annotations

from typing import Literal

# --- 机器人通信地址（直连 Mac，不经树莓派） ---
ROBOT_HOST: str = "192.168.137.100"
WEBSOCKET_URL: str = f"ws://{ROBOT_HOST}:8000"
VIDEO_STREAM_URL: str = f"http://{ROBOT_HOST}:8080/?action=stream"

# --- 视频采集 ---
# 读取帧超时（秒）。超过该时间未收到新帧视为断流。
FRAME_TIMEOUT_S: float = 3.0
# 断流后自动重连间隔（秒）
RECONNECT_DELAY_S: float = 2.0
# FPS 统计窗口大小
FPS_WINDOW: int = 30

# --- 窗口显示 ---
WINDOW_NAME: str = "ROV Camera"

# ============================================================
# 阶段 2：颜色识别
# ============================================================

# 目标颜色："green" 或 "red"，只选一种
TargetColor = Literal["green", "red"]
TARGET_COLOR: TargetColor = "green"

# HSV 阈值范围（OpenCV H: 0-179, S: 0-255, V: 0-255）
# 这是通用起始值，务必用 color_debug.py 按真实水下画面调整。
HSV_RANGES: dict[str, dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "green": {
        # 绿色在 HSV 中的常规范围
        "lower": (35, 80, 80),
        "upper": (85, 255, 255),
    },
    "red": {
        # 红色跨越 H=0/180 边界，检测时需要合并两段（见 detector.py）
        "lower": (0, 100, 80),
        "upper": (10, 255, 255),
    },
    "red_wrap": {
        # 红色第二段（靠近 H=179）
        "lower": (170, 100, 80),
        "upper": (179, 255, 255),
    },
}

# 轮廓过滤参数
MIN_CONTOUR_AREA: float = 300.0     # 小于此面积的轮廓视为噪声
MIN_CIRCULARITY: float = 0.55       # 圆形度阈值 4πA/P²，完美圆=1.0
MAX_ASPECT_RATIO: float = 1.8       # 宽高比超过此值视为非球形
