"""rov.detector 单元测试（离线运行，不需要摄像头/网络）。"""

from __future__ import annotations

import numpy as np
import pytest

from rov.detector import BallDetector, DetectionStatus


# ------------------------------------------------------------------ #
# 辅助：构造合成测试帧
# ------------------------------------------------------------------ #


def make_frame(width: int = 320, height: int = 240, color: tuple = (0, 0, 0)) -> np.ndarray:
    """创建指定 BGR 底色的空帧。"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def draw_bgr_circle(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int,
    bgr: tuple[int, int, int],
) -> np.ndarray:
    """在帧上画一个实心圆。"""
    import cv2

    cv2.circle(frame, center, radius, bgr, -1)
    return frame


def draw_bgr_rect(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    bgr: tuple[int, int, int],
) -> np.ndarray:
    import cv2

    cv2.rectangle(frame, top_left, bottom_right, bgr, -1)
    return frame


# OpenCV BGR：绿球 ≈ (0, 200, 0)，红球 ≈ (0, 0, 200)
GREEN_BGR = (0, 200, 0)
RED_BGR = (0, 0, 200)


# ------------------------------------------------------------------ #
# 绿球检测
# ------------------------------------------------------------------ #


class TestGreenDetection:
    def test_detects_green_ball(self) -> None:
        frame = make_frame()
        draw_bgr_circle(frame, (160, 120), 40, GREEN_BGR)

        detector = BallDetector("green")
        result = detector.detect(frame)

        assert result.is_detected
        assert result.status is DetectionStatus.DETECTED
        assert result.center is not None
        cx, cy = result.center
        assert abs(cx - 160) < 5
        assert abs(cy - 120) < 5
        assert result.area is not None and result.area > 1000
        assert result.bbox is not None

    def test_lost_on_blank_frame(self) -> None:
        frame = make_frame()
        detector = BallDetector("green")
        result = detector.detect(frame)

        assert result.status is DetectionStatus.LOST
        assert result.center is None
        assert result.area is None
        assert result.bbox is None

    def test_lost_on_wrong_color(self) -> None:
        """绿色检测器不应找到红球。"""
        frame = make_frame()
        draw_bgr_circle(frame, (160, 120), 40, RED_BGR)

        detector = BallDetector("green")
        result = detector.detect(frame)

        assert result.status is DetectionStatus.LOST

    def test_filters_small_contour(self) -> None:
        """小面积噪声应被过滤。"""
        frame = make_frame()
        draw_bgr_circle(frame, (50, 50), 5, GREEN_BGR)  # 极小圆

        detector = BallDetector("green", min_area=500.0)
        result = detector.detect(frame)

        assert result.status is DetectionStatus.LOST

    def test_filters_non_circular(self) -> None:
        """长条形状应被过滤。"""
        frame = make_frame()
        draw_bgr_rect(frame, (30, 100), (280, 120), GREEN_BGR)  # 很长的矩形

        detector = BallDetector("green")
        result = detector.detect(frame)

        assert result.status is DetectionStatus.LOST

    def test_picks_largest_ball(self) -> None:
        """多个球时取面积最大的。"""
        frame = make_frame()
        draw_bgr_circle(frame, (60, 120), 20, GREEN_BGR)   # 小球
        draw_bgr_circle(frame, (240, 120), 50, GREEN_BGR)  # 大球

        detector = BallDetector("green")
        result = detector.detect(frame)

        assert result.is_detected
        assert result.center is not None
        cx, _ = result.center
        assert abs(cx - 240) < 5  # 中心在大球处


# ------------------------------------------------------------------ #
# 红球检测
# ------------------------------------------------------------------ #


class TestRedDetection:
    def test_detects_red_ball(self) -> None:
        frame = make_frame()
        draw_bgr_circle(frame, (160, 120), 40, RED_BGR)

        detector = BallDetector("red")
        result = detector.detect(frame)

        assert result.is_detected
        assert result.center is not None
        cx, cy = result.center
        assert abs(cx - 160) < 5
        assert abs(cy - 120) < 5

    def test_lost_on_green_ball(self) -> None:
        """红色检测器不应找到绿球。"""
        frame = make_frame()
        draw_bgr_circle(frame, (160, 120), 40, GREEN_BGR)

        detector = BallDetector("red")
        result = detector.detect(frame)

        assert result.status is DetectionStatus.LOST


# ------------------------------------------------------------------ #
# DetectionResult 行为
# ------------------------------------------------------------------ #


class TestDetectionResult:
    def test_lost_has_no_coordinates(self) -> None:
        from rov.detector import DetectionResult

        result = DetectionResult.lost()
        assert result.status is DetectionStatus.LOST
        assert result.center is None
        assert result.area is None
        assert result.bbox is None
        assert result.radius is None
        assert not result.is_detected

    def test_detected_has_coordinates(self) -> None:
        from rov.detector import DetectionResult

        result = DetectionResult.detected(
            center=(100.0, 200.0), area=5000.0, bbox=(80, 180, 40, 40), radius=40.0
        )
        assert result.is_detected
        assert result.center == (100.0, 200.0)
        assert result.area == 5000.0


# ------------------------------------------------------------------ #
# 边界情况
# ------------------------------------------------------------------ #


class TestEdgeCases:
    def test_empty_frame_returns_lost(self) -> None:
        detector = BallDetector("green")
        result = detector.detect(np.array([]))
        assert result.status is DetectionStatus.LOST

    def test_custom_hsv_range(self) -> None:
        """自定义 HSV 阈值应生效。"""
        frame = make_frame()
        draw_bgr_circle(frame, (160, 120), 40, GREEN_BGR)

        # 故意用完全不匹配的范围
        detector = BallDetector(
            "green",
            hsv_lower=(100, 255, 255),
            hsv_upper=(110, 255, 255),
        )
        result = detector.detect(frame)
        assert result.status is DetectionStatus.LOST

        # 用匹配的范围
        detector2 = BallDetector(
            "green",
            hsv_lower=(35, 80, 80),
            hsv_upper=(85, 255, 255),
        )
        result2 = detector2.detect(frame)
        assert result2.is_detected

    def test_invalid_color_raises(self) -> None:
        with pytest.raises(ValueError):
            BallDetector("blue")  # type: ignore[arg-type]
