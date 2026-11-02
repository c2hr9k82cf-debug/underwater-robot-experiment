"""颜色球检测模块。

职责：
- 用 HSV 颜色分割提取目标颜色区域
- 轮廓过滤（面积、圆形度、宽高比）
- 输出检测结果（中心、面积、边界框、状态）
- 在画面上标注检测框和中心点

目标丢失时状态为 LOST，**不返回过期坐标**。
本模块不包含任何电机/推进器控制逻辑。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from rov.config import (
    HSV_RANGES,
    MAX_ASPECT_RATIO,
    MIN_CONTOUR_AREA,
    MIN_HULL_FILL,
    TARGET_COLOR,
    WALL_HSV_LOWER,
    WALL_HSV_UPPER,
    TargetColor,
)

logger = logging.getLogger(__name__)


class DetectionStatus(Enum):
    """检测状态。"""

    DETECTED = "DETECTED"
    LOST = "LOST"


@dataclass(frozen=True)
class DetectionResult:
    """一次检测的输出。LOST 时坐标类字段均为 None。"""

    status: DetectionStatus
    center: Optional[Tuple[float, float]]  # (cx, cy) 像素坐标
    area: Optional[float]                  # 轮廓面积（像素²）
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    radius: Optional[float]                # 等效半径（像素）

    @classmethod
    def lost(cls) -> "DetectionResult":
        return cls(
            status=DetectionStatus.LOST,
            center=None,
            area=None,
            bbox=None,
            radius=None,
        )

    @classmethod
    def detected(
        cls,
        center: Tuple[float, float],
        area: float,
        bbox: Tuple[int, int, int, int],
        radius: float,
    ) -> "DetectionResult":
        return cls(
            status=DetectionStatus.DETECTED,
            center=center,
            area=area,
            bbox=bbox,
            radius=radius,
        )

    @property
    def is_detected(self) -> bool:
        return self.status is DetectionStatus.DETECTED


@dataclass(frozen=True)
class ContourCandidate:
    """通过过滤的候选轮廓（调试用）。"""

    center: Tuple[float, float]
    area: float
    bbox: Tuple[int, int, int, int]
    radius: float
    hull_fill: float


@dataclass(frozen=True)
class WallSight:
    """蓝色墙壁视野：画面中下部三列走廊的蓝色占比（0-1）。"""

    left: float
    front: float
    right: float


def detect_wall(
    frame: np.ndarray,
    hsv_lower: Tuple[int, int, int] = WALL_HSV_LOWER,
    hsv_upper: Tuple[int, int, int] = WALL_HSV_UPPER,
) -> WallSight:
    """检测画面中蓝色墙壁的占比（避障用）。

    把画面中下部（去掉上 1/4，水面本身偏蓝会干扰）分成
    左/中/右三列走廊，返回各列蓝色像素占比。
    前方（中央列）占比高 = 正对墙，避障转向用左右列判断空旷侧。
    """
    if frame is None or frame.size == 0:
        return WallSight(0.0, 0.0, 0.0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    h, w = mask.shape
    y0 = h // 4
    third = w // 3
    left = float(mask[y0:, :third].mean()) / 255.0
    front = float(mask[y0:, third : 2 * third].mean()) / 255.0
    right = float(mask[y0:, 2 * third :].mean()) / 255.0
    return WallSight(left=left, front=front, right=right)


class BallDetector:
    """基于 HSV 颜色分割的球检测器。

    形状过滤采用凸包填充率（面积/凸包面积）而非圆形度：
    浑水中球面有接缝、常被画框裁切，圆形度会低至 0.3 以下，
    但凸包填充率稳定在 0.9 以上；水下杂质则填充率低。
    """

    def __init__(
        self,
        color: TargetColor = TARGET_COLOR,
        *,
        hsv_lower: Optional[Tuple[int, int, int]] = None,
        hsv_upper: Optional[Tuple[int, int, int]] = None,
        min_area: float = MIN_CONTOUR_AREA,
        min_hull_fill: float = MIN_HULL_FILL,
        max_aspect_ratio: float = MAX_ASPECT_RATIO,
    ) -> None:
        self.color: TargetColor = color
        self.min_area = min_area
        self.min_hull_fill = min_hull_fill
        self.max_aspect_ratio = max_aspect_ratio

        # 允许覆盖默认阈值（调试工具用）
        if hsv_lower is not None and hsv_upper is not None:
            self._hsv_segments: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = [
                (hsv_lower, hsv_upper)
            ]
        elif color == "red":
            # 红色跨 H=0/180，需两段合并
            seg1 = HSV_RANGES["red"]
            seg2 = HSV_RANGES["red_wrap"]
            self._hsv_segments = [
                (seg1["lower"], seg1["upper"]),
                (seg2["lower"], seg2["upper"]),
            ]
        elif color == "green":
            seg = HSV_RANGES["green"]
            self._hsv_segments = [(seg["lower"], seg["upper"])]
        else:
            raise ValueError(f"不支持的目标颜色：{color!r}，请选择 'green' 或 'red'")

        logger.info("BallDetector 初始化：color=%s, segments=%d", color, len(self._hsv_segments))

    # ------------------------------------------------------------------ #
    # 核心检测
    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """检测帧中的目标球。

        返回 DetectionResult。未找到合格目标时返回 LOST（坐标为 None）。
        """
        if frame is None or frame.size == 0:
            logger.warning("detect() 收到空帧")
            return DetectionResult.lost()

        mask = self.build_mask(frame)
        candidates = self._find_candidates(mask)

        if not candidates:
            return DetectionResult.lost()

        # 取面积最大的候选
        best = max(candidates, key=lambda c: c.area)
        return DetectionResult.detected(
            center=best.center,
            area=best.area,
            bbox=best.bbox,
            radius=best.radius,
        )

    def detect_with_candidates(
        self, frame: np.ndarray
    ) -> Tuple[DetectionResult, List[ContourCandidate]]:
        """检测并返回所有通过过滤的候选（调试用）。"""
        mask = self.build_mask(frame)
        candidates = self._find_candidates(mask)

        if not candidates:
            return DetectionResult.lost(), []

        best = max(candidates, key=lambda c: c.area)
        result = DetectionResult.detected(
            center=best.center,
            area=best.area,
            bbox=best.bbox,
            radius=best.radius,
        )
        return result, candidates

    # ------------------------------------------------------------------ #
    # 掩膜与轮廓
    # ------------------------------------------------------------------ #

    def build_mask(self, frame: np.ndarray) -> np.ndarray:
        """生成二值 HSV 掩膜。红色自动合并两段。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self._hsv_segments:
            seg_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined = cv2.bitwise_or(combined, seg_mask)

        # 形态学去噪：开运算去小亮点，闭运算填洞并桥接球面接缝
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        return combined

    def _find_candidates(self, mask: np.ndarray) -> List[ContourCandidate]:
        """从二值掩膜中提取并过滤候选轮廓。"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[ContourCandidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            # 凸包填充率：球体（哪怕被裁切、带接缝）接近 1，杂质明显更低
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area <= 0:
                continue
            hull_fill = area / hull_area
            if hull_fill < self.min_hull_fill:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w / h, h / w) if w > 0 and h > 0 else float("inf")
            if aspect > self.max_aspect_ratio:
                continue

            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            radius = math.sqrt(area / math.pi)

            candidates.append(
                ContourCandidate(
                    center=(cx, cy),
                    area=area,
                    bbox=(x, y, w, h),
                    radius=radius,
                    hull_fill=hull_fill,
                )
            )

        return candidates

    # ------------------------------------------------------------------ #
    # 标注
    # ------------------------------------------------------------------ #

    @staticmethod
    def annotate(
        frame: np.ndarray,
        result: DetectionResult,
        *,
        draw_candidates: Optional[List[ContourCandidate]] = None,
    ) -> np.ndarray:
        """在帧上标注检测结果（原地绘制）。

        - DETECTED：绿色框 + 中心十字 + 信息文字
        - LOST：左上角红色 "LOST"
        """
        if draw_candidates:
            for cand in draw_candidates:
                x, y, w, h = cand.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

        if result.is_detected and result.center is not None and result.bbox is not None:
            cx, cy = int(result.center[0]), int(result.center[1])
            x, y, w, h = result.bbox

            # 边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 中心十字
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            # 圆形轮廓近似
            r = int(result.radius) if result.radius else 0
            if r > 0:
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 1)
            # 信息
            info = f"({cx},{cy}) area={result.area:.0f}"
            cv2.putText(
                frame, info, (x, max(y - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame, "LOST", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
            )

        return frame
