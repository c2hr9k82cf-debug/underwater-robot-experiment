"""rov.detector 单元测试（离线运行，不需要摄像头/网络）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rov.detector import BallDetector, DetectionStatus, WallSight, detect_wall


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

    def test_filters_low_hull_fill(self) -> None:
        """L 形（凸包填充率低）应被过滤。"""
        frame = make_frame()
        # L 形：竖条 + 横条，凸包填充率约 0.6，低于 0.85 阈值
        draw_bgr_rect(frame, (60, 40), (100, 180), GREEN_BGR)
        draw_bgr_rect(frame, (60, 140), (220, 180), GREEN_BGR)

        detector = BallDetector("green")
        result = detector.detect(frame)

        assert result.status is DetectionStatus.LOST

    def test_filters_thin_shape(self) -> None:
        """细长矩形应被宽高比过滤。"""
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


# ------------------------------------------------------------------ #
# 真实水下截图回归测试（浑水红球标定验证，禁止改动 tests/data/ 图片）
# ------------------------------------------------------------------ #

DATA_DIR = Path(__file__).parent / "data"


class TestUnderwaterRegression:
    """用真实浑水截图验证红球检测（离线，不需要机器人）。"""

    def _load(self, name: str):
        import cv2

        img = cv2.imread(str(DATA_DIR / name))
        assert img is not None, f"缺少测试图片：{name}"
        return img

    def test_near_ball_detected(self) -> None:
        """近距离红球：应检出，中心落在球体区域。"""
        img = self._load("underwater_ball_near.png")
        result = BallDetector("red").detect(img)

        assert result.is_detected
        assert result.center is not None
        cx, cy = result.center
        # 球体在画面左下（实测中心约 (354, 593)）
        assert 250 < cx < 460
        assert 500 < cy < 700
        assert result.area is not None and result.area > 50000

    def test_closeup_ball_detected(self) -> None:
        """特写红球（球面带接缝）：应检出，中心在画面中下部。"""
        img = self._load("underwater_ball_closeup.png")
        result = BallDetector("red").detect(img)

        assert result.is_detected
        assert result.center is not None
        cx, cy = result.center
        # 实测中心约 (225, 347)
        assert 120 < cx < 330
        assert 250 < cy < 430

    def test_far_murky_lost(self) -> None:
        """远景浑水（球物理不可见）：应 LOST，不误检。"""
        img = self._load("underwater_far_murky.png")
        result = BallDetector("red").detect(img)

        assert result.status is DetectionStatus.LOST
        assert result.center is None

    def test_background_crop_lost(self) -> None:
        """纯背景裁块：应 LOST。"""
        img = self._load("underwater_ball_near.png")
        crop = img[300:600, 900:1200]  # 右侧浑水背景

        result = BallDetector("red").detect(crop)
        assert result.status is DetectionStatus.LOST


class TestGreenBallRegression:
    """真实水下绿球截图回归（2026-11-09 绿球标定验证）。"""

    def test_green_closeup_detected(self) -> None:
        """绿球特写（带接缝、浑水）：应检出，中心在画面右侧。"""
        import cv2

        img = cv2.imread(str(DATA_DIR / "underwater_green_closeup.png"))
        assert img is not None, "缺少测试图片：underwater_green_closeup.png"
        result = BallDetector("green").detect(img)

        assert result.is_detected
        assert result.center is not None
        cx, cy = result.center
        # 球体占画面右侧大部（实测中心约 (818, 515)）
        assert cx > 600
        assert result.area is not None and result.area > 300000


def test_default_color_follows_config() -> None:
    """不指定颜色时跟随 config.TARGET_COLOR（main.py 启动时可选红/绿）。"""
    from rov import config

    old = config.TARGET_COLOR
    try:
        config.TARGET_COLOR = "green"
        assert BallDetector().color == "green"
        config.TARGET_COLOR = "red"
        assert BallDetector().color == "red"
    finally:
        config.TARGET_COLOR = old


# ------------------------------------------------------------------ #
# 蓝色墙壁检测（避障，2026-11-02 按现场截屏标定）
# ------------------------------------------------------------------ #

WALL_BGR = (255, 0, 0)  # OpenCV 蓝


class TestWallDetection:
    def test_wall_on_right(self) -> None:
        """墙在右列：right 高，left/front 低。"""
        frame = make_frame()
        frame[:, 214:] = WALL_BGR  # 右 1/3
        sight = detect_wall(frame)
        assert sight.right > 0.5
        assert sight.front < 0.1
        assert sight.left < 0.1

    def test_wall_in_front(self) -> None:
        """正对墙：front 高。"""
        frame = make_frame()
        frame[:, 107:214] = WALL_BGR  # 中央 1/3
        sight = detect_wall(frame)
        assert sight.front > 0.5

    def test_open_water_all_low(self) -> None:
        """开阔水（灰）：三列都低。"""
        sight = detect_wall(make_frame())
        assert sight.left < 0.05
        assert sight.front < 0.05
        assert sight.right < 0.05

    def test_red_ball_not_wall(self) -> None:
        """红球不能误判为墙。"""
        frame = make_frame()
        draw_bgr_circle(frame, (160, 120), 50, (0, 0, 255))
        sight = detect_wall(frame)
        assert sight.front < 0.05

    def test_empty_frame(self) -> None:
        sight = detect_wall(np.array([]))
        assert sight == WallSight(0.0, 0.0, 0.0)
