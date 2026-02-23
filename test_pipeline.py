"""
Automated tests for facemeasure's core analysis pipeline.

These tests verify the derived-metric functions (fWHR, eyebrow V-shape) and
the landmark visualisation helper using synthetic data so they can run without
a GPU, network access, or the full py-feat model weights.

Run:
    pytest test_pipeline.py -v
"""

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Import the functions under test from the main application module
# ---------------------------------------------------------------------------
from home import (
    calculate_fwhr,
    calculate_eyebrow_v_shape,
    draw_landmarks_on_image,
    analyze_image,
    AU_COLUMNS,
    EMOTION_COLUMNS,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic 68-point landmark data
# ---------------------------------------------------------------------------

def _make_landmark_dict(landmarks_68x2):
    """Convert a (68, 2) array into the dict format used by facemeasure."""
    d = {}
    for i in range(68):
        d[f"LM_{i}_X"] = float(landmarks_68x2[i, 0])
        d[f"LM_{i}_Y"] = float(landmarks_68x2[i, 1])
    return d


def _default_landmarks():
    """Return a realistic-ish synthetic set of 68 landmarks on a 256×256 image.

    Landmarks are placed roughly where they would be on a frontal face so
    that metrics like fWHR yield sensible positive values.
    """
    rng = np.random.RandomState(42)
    pts = rng.uniform(60, 200, size=(68, 2))

    # Anchor specific landmarks used by fWHR and eyebrow V-shape
    pts[0] = [50, 150]     # left jawline
    pts[16] = [200, 150]   # right jawline  → width = 150
    pts[27] = [128, 80]    # nasion (top of nose bridge)
    pts[51] = [128, 170]   # upper lip      → height = 90  → fWHR ≈ 1.667

    # Eyebrows: left (18-21) slopes up inward, right (22-25) mirrors
    pts[18] = [70, 90]
    pts[19] = [80, 85]
    pts[20] = [90, 80]
    pts[21] = [100, 75]
    pts[22] = [150, 75]
    pts[23] = [160, 80]
    pts[24] = [170, 85]
    pts[25] = [180, 90]

    return pts


# ---------------------------------------------------------------------------
# Tests — fWHR
# ---------------------------------------------------------------------------

class TestFWHR:
    """Tests for calculate_fwhr()."""

    def test_basic_calculation(self):
        """fWHR = width / height with known anchor points."""
        pts = _default_landmarks()
        d = _make_landmark_dict(pts)
        fwhr = calculate_fwhr(d)
        expected = 150.0 / 90.0  # |200 - 50| / |170 - 80|
        assert pytest.approx(fwhr, rel=1e-4) == expected

    def test_zero_height_returns_zero(self):
        """When LM_27 and LM_51 have the same Y, height is 0 → return 0."""
        pts = _default_landmarks()
        pts[27] = [128, 100]
        pts[51] = [128, 100]  # same Y → height = 0
        d = _make_landmark_dict(pts)
        assert calculate_fwhr(d) == 0

    def test_positive_for_normal_face(self):
        """fWHR should always be positive for a normal face layout."""
        pts = _default_landmarks()
        d = _make_landmark_dict(pts)
        assert calculate_fwhr(d) > 0

    def test_scale_changes_do_not_affect_ratio(self):
        """Scaling all coordinates by a constant should not change fWHR."""
        pts = _default_landmarks()
        d1 = _make_landmark_dict(pts)
        d2 = _make_landmark_dict(pts * 2.0)
        assert pytest.approx(calculate_fwhr(d1), rel=1e-4) == calculate_fwhr(d2)


# ---------------------------------------------------------------------------
# Tests — Eyebrow V-shape
# ---------------------------------------------------------------------------

class TestEyebrowVShape:
    """Tests for calculate_eyebrow_v_shape()."""

    def test_returns_float(self):
        pts = _default_landmarks()
        d = _make_landmark_dict(pts)
        result = calculate_eyebrow_v_shape(d)
        assert isinstance(result, float)

    def test_symmetric_eyebrows_near_zero(self):
        """Perfectly symmetric, flat eyebrows should yield V ≈ 0."""
        pts = _default_landmarks()
        # Make both eyebrows perfectly flat at the same y
        for i in range(18, 22):
            pts[i] = [70 + (i - 18) * 10, 80]
        for i in range(22, 26):
            pts[i] = [150 + (i - 22) * 10, 80]
        d = _make_landmark_dict(pts)
        v = calculate_eyebrow_v_shape(d)
        assert abs(v) < 0.5  # approximately zero

    def test_v_shaped_eyebrows_nonzero(self):
        """Asymmetric eyebrows should yield non-zero V value."""
        pts = _default_landmarks()
        # Left eyebrow: slopes up toward centre (Y decreases toward medial end)
        pts[18] = [70, 100]
        pts[19] = [80, 90]
        pts[20] = [90, 80]
        pts[21] = [100, 70]
        # Right eyebrow: mirrors (slopes down toward centre from right)
        pts[22] = [150, 70]
        pts[23] = [160, 80]
        pts[24] = [170, 90]
        pts[25] = [180, 100]
        d = _make_landmark_dict(pts)
        v = calculate_eyebrow_v_shape(d)
        # The metric should detect asymmetry (non-zero)
        assert abs(v) > 0.5


# ---------------------------------------------------------------------------
# Tests — Landmark visualisation
# ---------------------------------------------------------------------------

class TestDrawLandmarks:
    """Tests for draw_landmarks_on_image()."""

    def test_returns_image_same_size(self):
        """Output image should have the same dimensions as the input."""
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        pts = _default_landmarks()
        d = _make_landmark_dict(pts)
        result = draw_landmarks_on_image(img, d)
        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_does_not_modify_original(self):
        """The original image should not be mutated."""
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        original_data = list(img.getdata())
        pts = _default_landmarks()
        d = _make_landmark_dict(pts)
        draw_landmarks_on_image(img, d)
        assert list(img.getdata()) == original_data

    def test_handles_missing_landmarks(self):
        """Should not crash if some landmark keys are missing."""
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        partial = {"LM_0_X": 50, "LM_0_Y": 50, "LM_1_X": 60, "LM_1_Y": 60}
        result = draw_landmarks_on_image(img, partial)
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# Tests — analyze_image with mocked detector
# ---------------------------------------------------------------------------

class _MockDetector:
    """Minimal mock of py-feat Detector for unit testing."""

    def __init__(self, return_faces=True, return_landmarks=True):
        self._return_faces = return_faces
        self._return_landmarks = return_landmarks

    def detect_faces(self, img_array):
        if not self._return_faces:
            return [[]]
        # Return one bounding box [x1, y1, x2, y2, confidence]
        return [[[10, 10, 200, 200, 0.99]]]

    def detect_landmarks(self, img_array, detected_faces=None):
        if not self._return_landmarks:
            return [[]]
        pts = _default_landmarks()
        return [[pts.tolist()]]

    def detect_aus(self, img_array, landmarks):
        return [[np.zeros(20).tolist()]]

    def detect_emotions(self, img_array, faces, landmarks):
        return [[np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).tolist()]]

    def detect_facepose(self, img_array, landmarks):
        return {"poses": [[[5.0, 2.0, -1.0]]]}


class TestAnalyzeImage:
    """Tests for analyze_image() using a mock detector."""

    def _make_img(self):
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def test_basic_landmarks_and_metrics(self):
        """Without optional flags, result should have landmarks + fWHR + Eyebrow_V."""
        det = _MockDetector()
        result = analyze_image(det, self._make_img())
        assert "LM_0_X" in result
        assert "LM_67_Y" in result
        assert "fWHR" in result
        assert "Eyebrow_V" in result
        assert "Error" not in result

    def test_no_face_returns_error(self):
        det = _MockDetector(return_faces=False)
        result = analyze_image(det, self._make_img())
        assert "Error" in result

    def test_no_landmarks_returns_error(self):
        det = _MockDetector(return_landmarks=False)
        result = analyze_image(det, self._make_img())
        assert "Error" in result

    def test_aus_flag(self):
        det = _MockDetector()
        result = analyze_image(det, self._make_img(), detect_aus=True)
        for col in AU_COLUMNS:
            assert col in result

    def test_emotions_flag(self):
        det = _MockDetector()
        result = analyze_image(det, self._make_img(), detect_emotions=True)
        for col in EMOTION_COLUMNS:
            assert col in result

    def test_pose_flag(self):
        det = _MockDetector()
        result = analyze_image(det, self._make_img(), detect_pose=True)
        assert "Pitch" in result
        assert "Roll" in result
        assert "Yaw" in result

    def test_all_flags_combined(self):
        det = _MockDetector()
        result = analyze_image(
            det, self._make_img(),
            detect_aus=True, detect_emotions=True, detect_pose=True,
        )
        assert "fWHR" in result
        assert "AU01" in result
        assert "happiness" in result
        assert "Yaw" in result


# ---------------------------------------------------------------------------
# Tests — Constants sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    def test_au_columns_count(self):
        assert len(AU_COLUMNS) == 20

    def test_emotion_columns_count(self):
        assert len(EMOTION_COLUMNS) == 7
