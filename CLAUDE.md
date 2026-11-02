# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

水下机器人（ROV）自主寻找并轻触彩色球的本地测试项目。Mac 通过网线直连机器人（不经树莓派）。Python 3.11 + OpenCV，无框架，不是 git 仓库。

**安全约束（必须遵守）**：所有现有测试工具默认不控制推进器。后续阶段的硬件控制必须显式开启并经操作者确认，且需保留物理急停方式。不要在未经确认的情况下添加任何电机/推进器控制逻辑。

## 常用命令

```bash
source .venv/bin/activate          # 每次先激活虚拟环境
python -m pytest tests/ -v         # 运行全部单元测试（离线，不需要机器人）
python -m pytest tests/test_detector.py::测试函数名 -v   # 运行单个测试
```

无需构建和 lint 步骤。依赖在 `requirements.txt`（opencv-python、numpy、websockets、pytest），装进 `.venv/`。

### 运行入口

| 命令 | 用途 |
|------|------|
| `python server.py` | 本机 WebSocket 模拟服务器（127.0.0.1:8000） |
| `python video_test.py` | 阶段 1：视频采集测试（`--reconnect` 断流重连，`-v` 调试日志） |
| `python color_tune.py` | 阶段 2：HSV 阈值键盘调节（推荐） |
| `python analyze_color.py <图片> red\|green` | 阶段 2：截图自动分析，输出推荐 HSV 范围 |
| `python color_debug.py [--image X \| --video Y] [--color green\|red]` | 阶段 2：滑块调节（macOS 上不好用，备选） |

## 架构

两部分：`rov/` 共享库 + 顶层可执行脚本。脚本都是薄入口，核心逻辑在库中。

- `rov/config.py` — 所有共享常量的唯一来源：机器人地址（`192.168.137.100`，WebSocket :8000，MJPEG 视频流 :8080）、视频参数、`TARGET_COLOR`（"green"/"red" 二选一）、`HSV_RANGES` 阈值、轮廓过滤参数（面积/圆形度/宽高比）。调参后统一写回这里。
- `rov/video_stream.py` — `VideoStream`：对 `cv2.VideoCapture` 的薄封装，FPS 统计、断流超时检测（`FRAME_TIMEOUT_S`）、可选自动重连。
- `rov/detector.py` — `BallDetector`：HSV 分割 → 形态学去噪 → 轮廓过滤（面积、圆形度 4πA/P²、宽高比）→ 取面积最大候选。输出 `DetectionResult`。

### 关键设计约定

- **LOST 语义**：检测失败时 `DetectionResult` 的坐标类字段（center/area/bbox/radius）全部为 None，绝不返回过期坐标。改动检测输出时保持这一约定。
- **红色双段 HSV**：红色跨 H=0/180 边界，`HSV_RANGES` 中 `red` + `red_wrap` 两段由 `detector.build_mask()` 自动 OR 合并。不要把它简化成单段。
- `BallDetector` 构造函数接受 `hsv_lower`/`hsv_upper` 覆盖默认阈值，调试工具（color_tune.py 等）用它做实时预览，不写回 config。

## server.py 协议

模拟机器人端，不是真实控制程序。消息为冒号分隔、恰好 5 段：`<p0>:<p1>:<p2>:<p3>:<flag>`，`flag=0` 普通控制，`flag=1` 关机信号；段数不对记为格式错误但仍有回复。服务端总是回复模拟传感器数据 `10:20:30`。

- `ping_interval=None` 在 `websockets.serve()` 中是有意为之——真实客户端不响应 ping。**不要"修复"它。**
- 绑定 `127.0.0.1` 而非 `0.0.0.0`，保持仅本机访问。
- 无真实硬件、无持久化、无鉴权，未经要求不要添加。
- print 输出为中文，新增日志保持此风格（`rov/` 库内用 `logging`，同样中文文案）。

## 测试

`tests/test_detector.py` 用 numpy 合成帧（画圆/矩形）离线测试 `BallDetector`，不需要摄像头或网络。新增检测逻辑时沿用合成帧模式，不要在测试中依赖真实视频流。
