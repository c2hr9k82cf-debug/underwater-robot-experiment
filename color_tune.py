#!/usr/bin/env python3
"""HSV 阈值键盘调节工具（简化版）。

用键盘按键实时调整 HSV 范围，预览检测效果。
比滑块更好用，特别适合 macOS。

按键说明：
  q        退出
  w/s      H Low  +5 / -5
  e/d      S Low  +5 / -5
  r/f      V Low  +5 / -5
  t/g      H High +5 / -5
  y/h      S High +5 / -5
  u/j      V High +5 / -5
  1/2      调整步长（1=粗调，2=细调）
  p        打印当前 HSV 阈值
  z        重置为默认值
  b/n      切换绿球/红球

本程序不会发送任何电机/推进器控制指令。
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from rov.config import VIDEO_STREAM_URL, HSV_RANGES
from rov.detector import BallDetector


def main() -> int:
    print("""
╔══════════════════════════════════════════════════════╗
║           HSV 阈值键盘调节工具                        ║
╠══════════════════════════════════════════════════════╣
║  按键说明：                                          ║
║    q       退出                                      ║
║    w/s     H Low  +5 / -5   （色相下限）             ║
║    e/d     S Low  +5 / -5   （饱和度下限）           ║
║    r/f     V Low  +5 / -5   （亮度下限）             ║
║    t/g     H High +5 / -5   （色相上限）             ║
║    y/h     S High +5 / -5   （饱和度上限）           ║
║    u/j     V High +5 / -5   （亮度上限）             ║
║    1/2     调整步长（1=粗调，2=细调）                ║
║    p       打印当前 HSV 阈值                        ║
║    z       重置为默认值                             ║
║    b/n     切换绿球/红球                           ║
╚══════════════════════════════════════════════════════╝
""")

    # 连接视频流
    print(f"正在连接视频流：{VIDEO_STREAM_URL}")
    cap = cv2.VideoCapture(VIDEO_STREAM_URL)

    if not cap.isOpened():
        print("❌ 无法打开视频流！请检查机器人连接。")
        return 1

    print("✅ 视频流已连接！\n")

    # 初始颜色和阈值
    current_color = "red"
    seg = HSV_RANGES[current_color.lower()]
    lower = list(seg["lower"])
    upper = list(seg["upper"])
    step = 5  # 调整步长

    # 创建窗口
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    print(f"当前颜色：{current_color}")
    print(f"当前阈值：H[{lower[0]}-{upper[0]}] S[{lower[1]}-{upper[1]}] V[{lower[2]}-{upper[2]}]")
    print(f"调整步长：{step}")
    print("\n请把球放在摄像头前，按 p 打印阈值，按 q 退出\n")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ 无法读取视频帧，尝试重连...")
            time.sleep(0.5)
            continue

        frame_count += 1

        # HSV 转换
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建掩膜
        lower_arr = np.array(lower, dtype=np.uint8)
        upper_arr = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_arr, upper_arr)

        # 形态学处理（去噪）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制检测结果
        result_frame = frame.copy()
        detected = False

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:  # 过滤小区域
                continue

            # 计算圆形度
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.5:  # 过滤非圆形
                    continue

            # 绘制检测框
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制中心点
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(result_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(result_frame, f"({cx},{cy})", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected = True

        # 显示状态
        if not detected:
            cv2.putText(result_frame, "LOST", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(result_frame, "DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示当前阈值
        info = f"{current_color} H[{lower[0]}-{upper[0]}] S[{lower[1]}-{upper[1]}] V[{lower[2]}-{upper[2]}] step={step}"
        cv2.putText(result_frame, info, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示图像
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result_frame)

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q 或 ESC 退出
            break

        elif key == ord('w'):  # H Low +
            lower[0] = min(179, lower[0] + step)
        elif key == ord('s'):  # H Low -
            lower[0] = max(0, lower[0] - step)

        elif key == ord('e'):  # S Low +
            lower[1] = min(255, lower[1] + step)
        elif key == ord('d'):  # S Low -
            lower[1] = max(0, lower[1] - step)

        elif key == ord('r'):  # V Low +
            lower[2] = min(255, lower[2] + step)
        elif key == ord('f'):  # V Low -
            lower[2] = max(0, lower[2] - step)

        elif key == ord('t'):  # H High +
            upper[0] = min(179, upper[0] + step)
        elif key == ord('g'):  # H High -
            upper[0] = max(0, upper[0] - step)

        elif key == ord('y'):  # S High +
            upper[1] = min(255, upper[1] + step)
        elif key == ord('h'):  # S High -
            upper[1] = max(0, upper[1] - step)

        elif key == ord('u'):  # V High +
            upper[2] = min(255, upper[2] + step)
        elif key == ord('j'):  # V High -
            upper[2] = max(0, upper[2] - step)

        elif key == ord('1'):  # 粗调
            step = 10
            print(f"调整步长：{step}（粗调）")
        elif key == ord('2'):  # 细调
            step = 1
            print(f"调整步长：{step}（细调）")

        elif key == ord('p'):  # 打印阈值
            print(f"\n{'='*60}")
            print(f"当前颜色：{current_color}")
            print(f"HSV 下限 (lower): ({lower[0]}, {lower[1]}, {lower[2]})")
            print(f"HSV 上限 (upper): ({upper[0]}, {upper[1]}, {upper[2]})")
            print(f"\n复制到 rov/config.py 的 HSV_RANGES：")
            print(f'    "{current_color.lower()}": {{')
            print(f'        "lower": ({lower[0]}, {lower[1]}, {lower[2]}),')
            print(f'        "upper": ({upper[0]}, {upper[1]}, {upper[2]}),')
            print(f'    }},')
            print(f"{'='*60}\n")

        elif key == ord('z'):  # 重置
            seg = HSV_RANGES[current_color.lower()]
            lower = list(seg["lower"])
            upper = list(seg["upper"])
            print("✓ 已重置为默认阈值")

        elif key == ord('b'):  # 切换到绿球
            current_color = "green"
            seg = HSV_RANGES["green"]
            lower = list(seg["lower"])
            upper = list(seg["upper"])
            print(f"✓ 切换到绿球检测")

        elif key == ord('n'):  # 切换到红球
            current_color = "red"
            seg = HSV_RANGES["red"]
            lower = list(seg["lower"])
            upper = list(seg["upper"])
            print(f"✓ 切换到红球检测")

    # 清理
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ 已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
