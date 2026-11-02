# ROV 本地控制测试

水下机器人自主寻找并轻触彩色球项目。Mac 通过网线直连机器人，不使用树莓派。

## 当前进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 视频采集 | ✅ 完成 |
| 2 | 颜色识别 | ✅ 完成 |
| 3 | 通信与推进器标定 | 待开始 |
| 4 | 闭环对准 | 待开始 |
| 5 | 接近与轻触 | 待开始 |

## 已确认的机器人地址

| 项目 | 地址 |
|------|------|
| 机器人主机 | `192.168.137.100` |
| WebSocket | `ws://192.168.137.100:8000` |
| 视频流 | `http://192.168.137.100:8080/?action=stream` |

## 环境安装

```bash
cd ~/rov-local-test
python3 -m venv .venv          # 已创建可跳过
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest             # 运行测试需要
```

## 阶段 1：视频采集测试

```bash
python video_test.py              # 显示画面 + FPS，按 q 退出
python video_test.py --reconnect  # 断流自动重连
python video_test.py -v           # 调试日志
```

## 阶段 2：颜色识别

### 2a. 运行单元测试（离线，不需要机器人）

```bash
python -m pytest tests/ -v
```

### 2b. HSV 阈值调试

先用调试工具根据真实水下画面调整颜色阈值：

```bash
# 实时视频流
python color_debug.py

# 本地图片（离线调试）
python color_debug.py --image path/to/frame.jpg

# 本地视频
python color_debug.py --video path/to/clip.mp4

# 指定颜色
python color_debug.py --color green   # 或 red
```

**操作方式：**
- 拖动滑块调整 H/S/V 下限和上限
- 按 `s` 打印当前阈值（复制到 `rov/config.py` 的 `HSV_RANGES`）
- 按 `g` / `r` 切换绿球/红球
- 按 `q` 退出

### 2c. 目标选择

编辑 `rov/config.py`：

```python
TARGET_COLOR: TargetColor = "green"   # 改为 "red" 可切换
```

### 检测输出说明

| 字段 | 含义 |
|------|------|
| `status` | `DETECTED` / `LOST` |
| `center` | 目标中心像素坐标 `(cx, cy)`，LOST 时为 None |
| `area` | 轮廓面积（像素²） |
| `bbox` | 边界框 `(x, y, w, h)` |
| `radius` | 等效半径（像素） |

**LOST 时坐标字段全部为 None**，不会返回过期坐标。

## 安全须知

- 当前所有测试程序 **默认不控制推进器**。
- 后续阶段的硬件控制必须显式开启，并经操作者确认。
- 真实运动测试前请保留可立即切断动力的物理急停方式。
- `server.py` 是本机 WebSocket **模拟服务器**，不是真实机器人控制程序。

## 项目结构

```
rov-local-test/
├── AGENTS.md            # Agent 工作指引
├── README.md            # 本文件
├── requirements.txt     # Python 依赖
├── server.py            # 本机 WebSocket 模拟服务器（勿删）
├── video_test.py        # 阶段 1 入口：视频采集测试
├── color_debug.py       # 阶段 2 入口：HSV 阈值调试工具
├── tests/
│   └── test_detector.py # 阶段 2 单元测试
└── rov/
    ├── __init__.py
    ├── config.py        # 共享配置（地址、HSV 阈值、检测参数）
    ├── video_stream.py  # MJPEG 读取模块
    └── detector.py      # 颜色球检测模块
```
