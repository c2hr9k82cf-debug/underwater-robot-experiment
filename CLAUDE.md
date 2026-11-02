# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

水下机器人（ROV）自主寻找并轻触红球的本地测试项目。Mac 通过网线直连机器人（不经树莓派），机器人地址 `192.168.137.100`。Python 3.11 + OpenCV，无框架。git 仓库，远程为 GitHub 公开仓库 `underwater-robot-experiment`（中文名「水下机器人实验」）。

**安全约束（必须遵守）**：默认模拟模式，不控制推进器。真实推进器输出必须显式开启并经操作者确认，且需保留物理急停方式。不要在未经确认的情况下添加任何电机/推进器控制逻辑。关机指令（标志 1）只允许从 `main.py` 菜单 7 发送。

## 常用命令

```bash
source .venv/bin/activate          # 每次先激活虚拟环境
python main.py                     # ★ 统一控制台入口（所有操作从数字菜单进）
python -m pytest tests/ -v         # 全部单元测试（离线，不需要机器人）
python -m pytest tests/test_detector.py::测试函数名 -v   # 单个测试
```

无需构建和 lint 步骤。依赖在 `requirements.txt`（opencv-python、numpy、websockets、pytest），装进 `.venv/`。

### 运行入口

**用户操作一律走 `main.py` 数字菜单**（视频测试/颜色标定/检测验证/手动控制/自主模式/系统检查/关闭机器人）。顶层脚本（video_test.py、color_tune.py、analyze_color.py、color_debug.py）保留作高级/离线用法。

| 命令 | 用途 |
|------|------|
| `python server.py` | 本机 WebSocket 模拟机器人（127.0.0.1:8000，供 comm 测试） |
| `python analyze_color.py <图片> red\|green` | 截图自动分析，输出推荐 HSV 范围 |
| `python color_tune.py` | HSV 阈值键盘调节 |
| `python video_test.py` | 视频采集测试（`--reconnect`、`-v`） |

## 架构

`rov/` 共享库 + 顶层薄入口脚本 + `rov/modes/` 界面层。

- `rov/config.py` — 所有共享常量的唯一来源：机器人地址、视频参数、`TARGET_COLOR`（当前 "red"）、`HSV_RANGES`（红球双段）、轮廓过滤参数（`MIN_CONTOUR_AREA`/`MIN_HULL_FILL`/`MAX_ASPECT_RATIO`）。调参后统一写回这里。
- `rov/thrusters.py` — PWM 映射与混控（纯函数）：`mix(surge, yaw, heave)` 反桨差速（`l = 1462 + (surge+yaw)×200`，`r = 1462 − (surge−yaw)×200`，竖直不反向），PWM 钳位 [1262, 1662]，`build_command`/`stop_command`/`shutdown_command`/`parse_sensor`。协议权威来源是 `机器人使用说明.pdf`（已入库）。
- `rov/comm.py` — `RobotComm`：websockets 客户端，**10 Hz 心跳发送循环**（防推进器鸣叫），推力超 0.5s 未更新自动回中位，断线重连，模拟/真实双模式。`live=False` 时不连接只记录 `last_command`。
- `rov/detector.py` — `BallDetector`：HSV 分割 → 形态学（open 5×5 + close 9×9）→ 轮廓过滤（面积、**凸包填充率 ≥0.85**、宽高比）→ 取面积最大候选。输出 `DetectionResult`。
- `rov/video_stream.py` — `VideoStream`：对 `cv2.VideoCapture` 的薄封装，FPS 统计、断流超时（`FRAME_TIMEOUT_S`）、可选自动重连。
- `rov/autonomy.py` — 自主状态机（**纯函数** `step(ctx, result, frame_size, now)`，无 IO，离线可测）：SEARCH→ALIGN→APPROACH→TOUCH→BACKOFF→DONE。参数集中在模块顶部。
- `rov/modes/` — 界面层：`manual.py`（WASD 驾驶）、`auto.py`（自主模式）、`detect.py`（检测验证）、`runtime.py`（`CommRunner` 跨线程通信桥、`draw_hud`、`save_snapshot`）。

### 关键设计约定

- **LOST 语义**：检测失败时 `DetectionResult` 的坐标类字段（center/area/bbox/radius）全部为 None，绝不返回过期坐标。改动检测输出时保持这一约定。
- **红色双段 HSV**：红色跨 H=0/180 边界，`HSV_RANGES` 中 `red` + `red_wrap` 两段由 `detector.build_mask()` 自动 OR 合并。不要把它简化成单段。
- **凸包填充率替代圆形度**：真实球面有接缝、会被画框裁切，圆形度实测仅 0.27–0.50 不可用；填充率 = 轮廓面积/凸包面积 ≥ 0.85。不要再改回圆形度过滤。
- **浑水预期 LOST**：远处球物理不可见，检测不到是正确行为；自主状态机靠 SEARCH 兜底、丢失带方向记忆。不要为了让远处检出而放松阈值。
- **手动模式松键回停**：按键保持 0.5s 无输入自动回中位（HOLD_S），是刻意的安全设计，不要改成"保持速度"。
- **心跳不能停**：`comm` 的 10 Hz 发送循环是防鸣叫保活，退出时补发 3 帧中位再断开。
- `BallDetector` 构造函数接受 `hsv_lower`/`hsv_upper` 覆盖默认阈值，调试工具用它实时预览，不写回 config。
- 终端输出不刷屏：状态画在视频 HUD，终端只打关键事件（连接/断流/状态切换/错误）。print 输出为中文（`rov/` 库内用 `logging`，同样中文文案）。
- 截图（`P` 键）保存到 `snapshots/`（原图，不带标注，便于 `analyze_color.py` 重新标定），不入库。

## server.py 协议

模拟机器人端，不是真实控制程序。消息为冒号分隔、恰好 5 段：`<p0>:<p1>:<p2>:<p3>:<flag>`，`flag=0` 普通控制，`flag=1` 关机信号；段数不对记为格式错误但仍有回复。服务端总是回复模拟传感器数据 `10:20:30`。

- `ping_interval=None` 在 `websockets.serve()` 中是有意为之——真实客户端不响应 ping。**不要"修复"它。**
- 绑定 `127.0.0.1` 而非 `0.0.0.0`，保持仅本机访问。
- 无真实硬件、无持久化、无鉴权，未经要求不要添加。

## 测试

- `tests/test_thrusters.py` — PWM 钳位、说明书前进示例（1523:1401）复现、转向符号、命令格式、传感器解析
- `tests/test_comm.py` — 自建 websockets mock 服务器测收发、心跳、过期回中位、停桨帧、关机指令
- `tests/test_detector.py` — numpy 合成帧 + `tests/data/` 真实水下截图回归（近距/特写 DETECTED、远景/背景 LOST）
- `tests/test_autonomy.py` — 状态机序列断言（全链路、方向记忆、丢失恢复、低置信只对中），注入假时间

新增检测逻辑时沿用合成帧模式，不要在测试中依赖真实视频流。新增状态机逻辑先写进 `autonomy.py` 纯函数并补测试，界面层只做 IO。
