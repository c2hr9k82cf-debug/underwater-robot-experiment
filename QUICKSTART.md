# 🚀 快速开始指南

## 📋 当前状态

✅ **阶段 1**：视频采集完成  
🔧 **阶段 2**：颜色识别 - 需要标定 HSV 阈值  
⏳ **阶段 3-5**：待开始  

---

## 🎯 下次要做什么

### 任务：标定 HSV 阈值

**目标**：找到适合你的球（红/绿）的 HSV 范围

**两种方法任选其一**：

---

### 方法 1️⃣：截图分析法（推荐 ⭐）

**优点**：自动分析，最准确，可离线操作

```bash
# 1. 进入项目
cd ~/rov-local-test
source .venv/bin/activate

# 2. 把球放在摄像头前，截图
python -c "import cv2, time; cap=cv2.VideoCapture('http://192.168.137.100:8080/?action=stream'); time.sleep(2); ret, frame=cap.read(); cv2.imwrite('ball_photo.jpg', frame) if ret else print('失败'); cap.release(); print('✅ 截图完成')"

# 3. 查看图片（确认球在里面）
open ball_photo.jpg

# 4. 分析颜色（红球）
python analyze_color.py ball_photo.jpg red

# 或分析绿球
python analyze_color.py ball_photo.jpg green
```

**分析脚本会输出**：
- 📊 检测到的颜色区域
- ✨ 推荐的 HSV 范围（直接复制到 config.py）
- 📸 可视化结果图片

---

### 方法 2️⃣：键盘调节法

**优点**：实时调整，直观

```bash
# 1. 进入项目
cd ~/rov-local-test
source .venv/bin/activate

# 2. 运行键盘调节工具
python color_tune.py
```

**按键说明**：

| 按键 | 功能 |
|------|------|
| `w/s` | H Low +5/-5（色相下限）|
| `e/d` | S Low +5/-5（饱和度下限）|
| `r/f` | V Low +5/-5（亮度下限）|
| `t/g` | H High +5/-5（色相上限）|
| `y/h` | S High +5/-5（饱和度上限）|
| `u/j` | V High +5/-5（亮度上限）|
| `1/2` | 粗调/细调切换 |
| `p` | 打印当前阈值 |
| `z` | 重置为默认值 |
| `b/n` | 切换绿球/红球 |
| `q` | 退出 |

**目标**：Mask 窗口中球是**白色**，背景是**黑色**

---

## 📊 完整测试流程

### 1️⃣ 测试视频（阶段 1）

```bash
python video_test.py
```

✅ 预期：显示视频 + FPS，按 `q` 退出

### 2️⃣ 标定颜色（阶段 2）

```bash
# 截图分析（推荐）
python analyze_color.py ball_photo.jpg red

# 或实时调节
python color_tune.py
```

✅ 预期：球被正确检测，输出 HSV 范围

### 3️⃣ 更新配置

将分析输出的 HSV 值复制到 `rov/config.py`：

```python
HSV_RANGES = {
    "red": {
        "lower": (0, 100, 100),     # ← 替换为分析输出的值
        "upper": (10, 255, 255),    # ← 替换为分析输出的值
    },
}
```

### 4️⃣ 验证检测

```bash
# 运行单元测试
python -m pytest tests/ -v

# 或用 color_tune.py 实时验证
python color_tune.py
```

---

## 🔧 常用命令速查

```bash
# 进入环境
cd ~/rov-local-test
source .venv/bin/activate

# 测试视频
python video_test.py

# 截图
python -c "import cv2, time; cap=cv2.VideoCapture('http://192.168.137.100:8080/?action=stream'); time.sleep(2); ret, frame=cap.read(); cv2.imwrite('ball_photo.jpg', frame) if ret else print('失败'); cap.release(); print('截图完成')"

# 分析颜色
python analyze_color.py ball_photo.jpg red

# 调节颜色
python color_tune.py

# 运行测试
python -m pytest tests/ -v

# 查看图片
open ball_photo.jpg
```

---

## ❓ 遇到问题？

### 视频打不开
```bash
# 检查网络
ping 192.168.137.100

# 重启机器人
# 检查网线连接
```

### 检测不到球
1. 确保球在摄像头前
2. 光线要充足
3. 用 analyze_color.py 自动分析
4. 水下测试需要重新标定（颜色会偏蓝）

### 滑块拖不动
使用 `color_tune.py`（键盘版）替代

---

## 📁 项目文件

| 文件 | 用途 |
|------|------|
| `video_test.py` | 测试视频 |
| `color_tune.py` | 键盘调节 HSV（推荐）|
| `analyze_color.py` | 自动分析颜色（推荐）|
| `color_debug.py` | 滑块调节 HSV（备用）|
| `rov/config.py` | 配置文件（HSV 阈值在这里）|
| `tests/` | 单元测试 |

---

## 🔒 安全提醒

⚠️ **所有测试工具都不会控制电机！**

- ✅ 只读取视频
- ✅ 只做图像处理
- ❌ 不发送控制指令
- ❌ 不控制推进器

**真实推进器测试需要阶段 3 之后，并且需要人工确认！**

---

## 📝 下次继续清单

- [ ] 截图（把球放在摄像头前）
- [ ] 分析颜色（得到 HSV 范围）
- [ ] 更新 config.py
- [ ] 验证检测效果
- [ ] （可选）下水重新标定

---

**准备好了吗？开始标定颜色吧！** 🎯
