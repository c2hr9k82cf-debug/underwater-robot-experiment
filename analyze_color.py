#!/usr/bin/env python3
"""自动分析图片中球的颜色并推荐 HSV 范围。

使用方法：
  1. 先截图：python -c "..."  (或使用 video_test.py)
  2. 运行本脚本：python analyze_color.py [图片路径]

本程序不会发送任何电机/推进器控制指令。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def analyze_red_ball(img_path: str) -> None:
    """分析红色球的 HSV 范围。"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图片：{img_path}")
        return

    print(f"📊 正在分析图片：{img_path}")
    print(f"   图像尺寸：{img.shape[1]}x{img.shape[0]}")
    print()

    # 转换到 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色范围 1 (H: 0-10)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    # 红色范围 2 (H: 170-179)
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 合并两个范围
    mask = mask1 + mask2

    # 统计红色像素
    red_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    percentage = (red_pixels / total_pixels) * 100

    print(f"🔴 检测到红色像素：{red_pixels} ({percentage:.2f}%)")

    if percentage < 0.1:
        print("⚠  未检测到足够的红色，请确保：")
        print("   - 球是红色的")
        print("   - 球在画面中清晰可见")
        print("   - 光线充足")
        return

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠  未找到轮廓")
        return

    # 找最大轮廓
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < 500:
        print(f"⚠  检测到的区域太小（{area:.0f} 像素），请把球放近一点")
        return

    print(f"✅ 找到红色区域，面积：{area:.0f} 像素")
    print()

    # 计算边界框
    x, y, w, h = cv2.boundingRect(largest)
    print(f"📍 位置：x={x}, y={y}, 宽={w}, 高={h}")
    print()

    # 提取该区域的 HSV 统计
    roi_hsv = hsv[y:y+h, x:x+w]
    h_vals = roi_hsv[:, :, 0].flatten()
    s_vals = roi_hsv[:, :, 1].flatten()
    v_vals = roi_hsv[:, :, 2].flatten()

    # 计算 5% 和 95% 分位数（排除异常值）
    h_min, h_max = np.percentile(h_vals, [5, 95])
    s_min, s_max = np.percentile(s_vals, [5, 95])
    v_min, v_max = np.percentile(v_vals, [5, 95])

    print("=" * 60)
    print("✨ 推荐的 HSV 范围：")
    print("=" * 60)
    print()
    print("复制下面的内容到 rov/config.py 的 HSV_RANGES：")
    print()
    print("    \"red\": {")
    print(f"        \"lower\": ({int(h_min)}, {int(s_min)}, {int(v_min)}),")
    print(f"        \"upper\": ({int(h_max)}, {int(s_max)}, {int(v_max)}),")
    print("    },")
    print()
    print("=" * 60)
    print()

    # 保存可视化结果
    result = img.copy()
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.circle(result, (x+w//2, y+h//2), 5, (0, 0, 255), -1)

    output_dir = Path(img_path).parent
    cv2.imwrite(str(output_dir / "analysis_result.jpg"), result)
    cv2.imwrite(str(output_dir / "analysis_mask.jpg"), mask)

    print("📸 已保存分析结果：")
    print(f"   - {output_dir / 'analysis_result.jpg'} (检测框)")
    print(f"   - {output_dir / 'analysis_mask.jpg'} (黑白掩膜)")
    print()
    print("💡 提示：")
    print("   - 如果水下测试，请在水下截图后重新分析")
    print("   - 水下颜色会偏蓝，HSV 值会不同")


def analyze_green_ball(img_path: str) -> None:
    """分析绿色球的 HSV 范围。"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图片：{img_path}")
        return

    print(f"📊 正在分析图片：{img_path}")
    print(f"   图像尺寸：{img.shape[1]}x{img.shape[0]}")
    print()

    # 转换到 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 绿色范围
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 统计绿色像素
    green_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    percentage = (green_pixels / total_pixels) * 100

    print(f"🟢 检测到绿色像素：{green_pixels} ({percentage:.2f}%)")

    if percentage < 0.1:
        print("⚠  未检测到足够的绿色，请确保：")
        print("   - 球是绿色的")
        print("   - 球在画面中清晰可见")
        print("   - 光线充足")
        return

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠  未找到轮廓")
        return

    # 找最大轮廓
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < 500:
        print(f"⚠  检测到的区域太小（{area:.0f} 像素），请把球放近一点")
        return

    print(f"✅ 找到绿色区域，面积：{area:.0f} 像素")
    print()

    # 计算边界框
    x, y, w, h = cv2.boundingRect(largest)
    print(f"📍 位置：x={x}, y={y}, 宽={w}, 高={h}")
    print()

    # 提取该区域的 HSV 统计
    roi_hsv = hsv[y:y+h, x:x+w]
    h_vals = roi_hsv[:, :, 0].flatten()
    s_vals = roi_hsv[:, :, 1].flatten()
    v_vals = roi_hsv[:, :, 2].flatten()

    # 计算 5% 和 95% 分位数
    h_min, h_max = np.percentile(h_vals, [5, 95])
    s_min, s_max = np.percentile(s_vals, [5, 95])
    v_min, v_max = np.percentile(v_vals, [5, 95])

    print("=" * 60)
    print("✨ 推荐的 HSV 范围：")
    print("=" * 60)
    print()
    print("复制下面的内容到 rov/config.py 的 HSV_RANGES：")
    print()
    print("    \"green\": {")
    print(f"        \"lower\": ({int(h_min)}, {int(s_min)}, {int(v_min)}),")
    print(f"        \"upper\": ({int(h_max)}, {int(s_max)}, {int(v_max)}),")
    print("    },")
    print()
    print("=" * 60)
    print()

    # 保存可视化结果
    result = img.copy()
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.circle(result, (x+w//2, y+h//2), 5, (0, 0, 255), -1)

    output_dir = Path(img_path).parent
    cv2.imwrite(str(output_dir / "analysis_result.jpg"), result)
    cv2.imwrite(str(output_dir / "analysis_mask.jpg"), mask)

    print("📸 已保存分析结果：")
    print(f"   - {output_dir / 'analysis_result.jpg'} (检测框)")
    print(f"   - {output_dir / 'analysis_mask.jpg'} (黑白掩膜)")
    print()
    print("💡 提示：")
    print("   - 如果水下测试，请在水下截图后重新分析")
    print("   - 水下颜色会偏蓝，HSV 值会不同")


def main() -> int:
    if len(sys.argv) < 2:
        print("""
🔍 颜色分析工具

使用方法：
  python analyze_color.py <图片路径> [颜色]

参数：
  图片路径    要分析的图片文件
  颜色        red 或 green（默认：red）

示例：
  python analyze_color.py ball_photo.jpg red
  python analyze_color.py test_frame.jpg green

提示：
  1. 先把球放在摄像头前
  2. 截图：python -c "import cv2, time; cap=cv2.VideoCapture('http://192.168.137.100:8080/?action=stream'); time.sleep(2); ret, frame=cap.read(); cv2.imwrite('ball_photo.jpg', frame) if ret else print('失败'); cap.release(); print('截图完成')"
  3. 分析：python analyze_color.py ball_photo.jpg red
        """)
        return 0

    img_path = sys.argv[1]
    color = sys.argv[2] if len(sys.argv) > 2 else "red"

    if not Path(img_path).exists():
        print(f"❌ 文件不存在：{img_path}")
        return 1

    if color.lower() == "red":
        analyze_red_ball(img_path)
    elif color.lower() == "green":
        analyze_green_ball(img_path)
    else:
        print(f"❌ 不支持的颜色：{color}，请选择 red 或 green")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
