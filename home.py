"""
facemeasure — A web application for scalable facial metric extraction.

This Streamlit application provides a browser-based interface for extracting
facial landmarks, derived morphological metrics (fWHR, eyebrow V-shape),
action units (AUs), emotion classifications, and head pose estimates from
uploaded facial images.  It wraps the py-feat library (Cheong et al., 2023)
and exposes individual detection stages through user-facing toggles so that
researchers can skip expensive analyses they do not need.

Usage:
    streamlit run home.py

Repository: https://github.com/saurabh-khanna/facemeasure
License:    AGPL-3.0
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from streamlit_lottie import st_lottie
from datetime import datetime
import requests
import random
import json
import time
import torch
import inspect


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_lottie_url(url: str):
    """Fetch a Lottie animation JSON from *url*, cached across reruns."""
    try:
        r = requests.get(url, timeout=5)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    return r.json()

# ---------------------------------------------------------------------------
# Page configuration  (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_icon="👤",
    page_title="facemeasure",
    layout="centered",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    })

# ---------------------------------------------------------------------------
# Header & hero animation
# ---------------------------------------------------------------------------
st.title(":bust_in_silhouette: facemeasure")

# Centre the Lottie animation using a golden-ratio column layout
col1, col2, col3 = st.columns([1, 1.618, 1])

with col2:
    lottie_json = load_lottie_url("https://lottie.host/71c80b64-c8c4-41a8-a469-ad6ba3555abe/RK6dp4pBsY.json")
    if lottie_json:
        st_lottie(lottie_json, height=120)

# Introductory blurb with a blinking cursor animation
st.markdown("""
    <style>
    @keyframes blink {
        0%, 100% { opacity: 0; }
        50% { opacity: 1; }
    }
    .blinking-underscore {
        animation: blink 1.5s step-start infinite;
    }

    .block-container {
        padding-top: 2.25rem;
        padding-bottom: 2.25rem;
    }

    .block-container h1 {
        margin-bottom: 0.15rem;
        padding-bottom: 0;
        line-height: 1.2;
    }

    .block-container h3 {
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
    }

    [data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }

    [data-testid="stExpander"] {
        margin-bottom: 0;
    }

    [data-testid="stExpander"] details {
        padding-bottom: 0;
    }

    [data-testid="stExpander"] summary {
        font-size: 0.78rem;
        min-height: 1.9rem;
    }

    [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
        gap: 0.25rem;
    }

    [data-testid="stExpander"] [data-testid="column"] {
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }

    [data-testid="stExpander"] [data-testid="stToggle"] {
        min-height: 1.45rem;
    }

    [data-testid="stExpander"] [data-testid="stToggle"] label,
    [data-testid="stExpander"] [data-testid="stToggle"] p,
    [data-testid="stExpander"] [data-testid="stCaptionContainer"] {
        font-size: 0.68rem;
        line-height: 1.15;
    }

    [data-testid="stExpander"] [data-testid="stToggle"] [role="switch"] {
        transform: scale(0.78);
        transform-origin: left center;
    }

    [data-testid="stForm"] {
        border: 0;
        padding: 0;
    }

    [data-testid="stDataFrame"] {
        max-height: 24vh;
        overflow: auto;
    }

    [data-testid="stImage"] img {
        max-height: 24vh;
        object-fit: contain;
    }

    .legal-footer {
        position: fixed;
        right: 1rem;
        bottom: 0.45rem;
        left: 1rem;
        z-index: 999;
        color: #5f6b73;
        font-size: 0.68rem;
        line-height: 1.25;
        text-align: center;
    }

    .legal-footer a {
        color: #3d5968;
        text-decoration: none;
    }

    .legal-footer a:hover {
        text-decoration: underline;
    }
    </style>
    <p style='text-align: left; font-size: 16px; line-height: 1.35; margin: 0 0 1.1rem 0;'>
        <b>FaceMeasure</b> democratizes facial analysis by enabling researchers
        to instantly extract precise facial metrics - no expensive software or
        programming required. Upload one or more facial images to begin<b><span class="blinking-underscore">_</span></b>
    </p>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — project information and privacy notice
# ---------------------------------------------------------------------------
st.sidebar.title(":bust_in_silhouette: facemeasure")

st.sidebar.markdown("""
<div style="
    background-color: #DAE1E5;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
">
    <p>🌱 Supported by the Social and Behavioural Data Science Centre, University of Amsterdam</p>
    <p>🙈 No identifiable data is stored. Everything is cleared once you refresh this page or download results.</p>
    <p>🐧 Our open-source code base is available on <a href="https://github.com/saurabh-khanna/facemeasure" target="_blank">GitHub</a>.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="legal-footer">
    For more information contact Zak Witkower
    (<a href="mailto:zakwitkower@gmail.com">zakwitkower@gmail.com</a>;
    <a href="https://zakwitkower.com" target="_blank">zakwitkower.com</a>)
    or Saurabh Khanna
    (<a href="mailto:s.khanna@uva.nl">s.khanna@uva.nl</a>;
    <a href="https://saurabh-khanna.github.io/" target="_blank">saurabh-khanna.github.io</a>).
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading — cached across Streamlit reruns via @st.cache_resource
#
# py-feat loads ALL model weights eagerly at __init__ (~1 GB total).
# We pay this cost once; subsequent reruns reuse the cached Detector.
# The selective speed-up happens at *inference* time, not at init: we call
# individual detect_* methods instead of the monolithic detect_image().
#
# Model choices:
#   face_model      = 'retinaface'     (small CNN, 1.7 MB)
#   landmark_model  = 'mobilefacenet'   (12 MB, batch-safe)
#   au_model        = 'xgb'            (returns AU probabilities)
#   emotion_model   = 'svm'            (fast; avoids 529 MB resmasknet)
#   facepose_model  = 'img2pose'       (Euler angles: pitch, roll, yaw)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading face detection models...")
def load_detector():
    """Initialise and return a cached py-feat Detector instance."""
    # py-feat 0.6.x imports nltools, which still expects SciPy's old
    # binom_test helper. Modern SciPy exposes the same behavior as binomtest.
    try:
        import scipy.stats as scipy_stats
        if not hasattr(scipy_stats, "binom_test") and hasattr(scipy_stats, "binomtest"):
            def _binom_test(x, n=None, p=0.5, alternative="two-sided"):
                return scipy_stats.binomtest(x, n=n, p=p, alternative=alternative).pvalue
            scipy_stats.binom_test = _binom_test

        import scipy.integrate as scipy_integrate
        if not hasattr(scipy_integrate, "simps") and hasattr(scipy_integrate, "simpson"):
            def _simps(y, x=None, dx=1.0, axis=-1, even=None):
                return scipy_integrate.simpson(y, x=x, dx=dx, axis=axis)
            scipy_integrate.simps = _simps
    except Exception:
        pass

    try:
        import torchvision.io as torchvision_io
        if not hasattr(torchvision_io, "read_video"):
            def _read_video_unavailable(*args, **kwargs):
                raise RuntimeError("Video input is unavailable in this local torchvision build.")
            torchvision_io.read_video = _read_video_unavailable
    except Exception:
        pass

    from feat import Detector
    return Detector(
        face_model='retinaface',
        landmark_model='mobilefacenet',
        au_model='xgb',
        emotion_model='svm',
        facepose_model='img2pose',
        device='cpu',
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 20 Facial Action Units returned by py-feat's XGB model (Ekman & Friesen, 1978)
AU_COLUMNS = [
    'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10',
    'AU11', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24',
    'AU25', 'AU26', 'AU28', 'AU43',
]

# Basic emotion categories from py-feat's SVM classifier
EMOTION_COLUMNS = [
    'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral',
]

# Landmark connection topology for visualisation.  The 68-point markup
# follows the iBUG 300-W convention.  Groups of consecutive landmark
# indices are connected with line segments; "closed" groups (eyes, lips)
# have an additional segment from the last point back to the first.
_LANDMARK_GROUPS = [
    list(range(0, 17)),      # 0  Jawline
    list(range(17, 22)),     # 1  Left eyebrow
    list(range(22, 27)),     # 2  Right eyebrow
    list(range(27, 31)),     # 3  Nose bridge
    list(range(31, 36)),     # 4  Lower nose
    list(range(36, 42)),     # 5  Left eye
    list(range(42, 48)),     # 6  Right eye
    list(range(48, 60)),     # 7  Outer lip
    list(range(60, 68)),     # 8  Inner lip
]
_CLOSED_GROUPS = {4, 5, 6, 7, 8}  # lower nose + eyes + lips are closed loops


# ---------------------------------------------------------------------------
# Derived facial metrics — computed from 68 landmark coordinates
# ---------------------------------------------------------------------------

def calculate_eyebrow_v_shape(landmarks_dict):
    """Compute an eyebrow V-shape index from landmark slopes.

    The metric captures the degree to which the medial ends of the eyebrows
    are raised relative to the lateral ends, producing a V- or inverted-V
    appearance.  Landmark coordinates are first z-standardised across all 
    68 points so the result is scale-invariant.

    Args:
        landmarks_dict: dict mapping 'LM_{i}_X' / 'LM_{i}_Y' to floats.

    Returns:
        float: Positive values indicate a V (inner ends raised), negative
        values indicate an inverted-V (inner ends lowered).
    """
    left_eyebrow = [(landmarks_dict[f"LM_{i}_X"], landmarks_dict[f"LM_{i}_Y"]) for i in range(18, 22)]
    right_eyebrow = [(landmarks_dict[f"LM_{i}_X"], landmarks_dict[f"LM_{i}_Y"]) for i in range(22, 26)]

    all_x = [landmarks_dict[f"LM_{i}_X"] for i in range(68)]
    all_y = [landmarks_dict[f"LM_{i}_Y"] for i in range(68)]
    x_mean, x_std = np.mean(all_x), np.std(all_x)
    y_mean, y_std = np.mean(all_y), np.std(all_y)

    def standardize_points(points):
        return [((x - x_mean) / x_std, (y - y_mean) / y_std) for x, y in points]

    left_std = standardize_points(left_eyebrow)
    right_std = standardize_points(right_eyebrow)

    def calculate_slope(points):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        n = len(points)
        slope = (n * sum(x * y for x, y in points) - sum(x_coords) * sum(y_coords)) / \
                (n * sum(x * x for x in x_coords) - sum(x_coords) ** 2)
        return slope

    left_slope = calculate_slope(left_std)
    right_slope = calculate_slope(right_std)

    right_slope_rc = -1 * right_slope
    eyebrow_v = (left_slope + right_slope_rc) / 2
    return eyebrow_v


def calculate_fwhr(landmarks_dict):
    """Compute the facial width-to-height ratio (fWHR).

    fWHR is calculated as bizygomatic width (distance between landmarks 0
    and 16, i.e. the jawline endpoints) divided by upper-face height
    (distance between landmark 27 at the nasion and landmark 51 at the
    upper lip) following Carré & McCormick (2008).

    Args:
        landmarks_dict: dict mapping 'LM_{i}_X' / 'LM_{i}_Y' to floats.

    Returns:
        float: The fWHR value, or 0 if the height is zero.
    """
    width = abs(landmarks_dict["LM_16_X"] - landmarks_dict["LM_0_X"])
    height = abs(landmarks_dict["LM_51_Y"] - landmarks_dict["LM_27_Y"])
    return width / height if height != 0 else 0


def procrustes_align_landmark_arrays(landmark_arrays, max_iterations=10, tolerance=1e-7):
    """Align 68-point landmark arrays with Generalized Procrustes Analysis.

    The returned coordinates are centered, scaled to unit Frobenius norm, and
    rotated to a common mean shape. Reflections are not allowed, preserving the
    left/right orientation of each face.
    """
    if len(landmark_arrays) == 0:
        return []

    shapes = []
    for landmarks in landmark_arrays:
        pts = np.asarray(landmarks, dtype=float)
        if pts.shape != (68, 2) or not np.isfinite(pts).all():
            raise ValueError("Each landmark array must have shape (68, 2) and finite values.")

        pts = pts - np.mean(pts, axis=0)
        scale = np.linalg.norm(pts)
        if scale == 0:
            raise ValueError("Cannot Procrustes-align landmarks with zero spread.")
        shapes.append(pts / scale)

    reference = shapes[0]
    aligned = shapes
    for _ in range(max_iterations):
        aligned = []
        for pts in shapes:
            u, _, vt = np.linalg.svd(pts.T @ reference)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vt
            aligned.append(pts @ rotation)

        new_reference = np.mean(aligned, axis=0)
        new_reference = new_reference - np.mean(new_reference, axis=0)
        ref_scale = np.linalg.norm(new_reference)
        if ref_scale == 0:
            break
        new_reference = new_reference / ref_scale

        if np.linalg.norm(new_reference - reference) < tolerance:
            reference = new_reference
            break
        reference = new_reference

    return aligned


def landmark_array_to_result(landmarks, result):
    """Write a (68, 2) landmark array into a result dict as LM_* columns."""
    for i in range(68):
        result[f"LM_{i}_X"] = round(float(landmarks[i, 0]), 4)
        result[f"LM_{i}_Y"] = round(float(landmarks[i, 1]), 4)


def apply_procrustes_to_results(results):
    """Replace valid result landmark columns with batch Procrustes coordinates."""
    valid_results = [r for r in results if "Error" not in r and "_Raw_Landmarks" in r]
    if not valid_results:
        return

    raw_landmarks = [np.asarray(r["_Raw_Landmarks"], dtype=float) for r in valid_results]
    aligned_landmarks = procrustes_align_landmark_arrays(raw_landmarks)

    for result, landmarks in zip(valid_results, aligned_landmarks):
        landmark_array_to_result(landmarks, result)
        try:
            result["Eyebrow_V"] = round(calculate_eyebrow_v_shape(result), 4)
            result["fWHR"] = round(calculate_fwhr(result), 4)
        except Exception:
            result["Eyebrow_V"] = None
            result["fWHR"] = None


# ---------------------------------------------------------------------------
# Core analysis pipeline
# ---------------------------------------------------------------------------

def analyze_image(detector, img_array, detect_aus=False, detect_emotions=False, detect_pose=False):
    """Run the facial analysis pipeline on a single image.

    Instead of using py-feat's monolithic ``detect_image()`` — which always
    executes every detection stage — this function calls individual
    ``detect_*`` methods selectively.  The mandatory stages (face detection
    and landmark detection) are always run because they are prerequisites
    for the derived metrics (fWHR, eyebrow V-shape).  Optional stages
    (AUs, emotions, head pose) are only executed when the corresponding
    flag is ``True``, saving substantial processing time per image.

    Args:
        detector:        A py-feat ``Detector`` instance.
        img_array:       NumPy array of shape (H, W, 3), dtype uint8.
        detect_aus:      If True, run Action Unit detection.
        detect_emotions: If True, run emotion classification.
        detect_pose:     If True, run head pose estimation.

    Returns:
        dict: Keys are column names (e.g. 'LM_0_X', 'fWHR', 'AU01', ...)
              mapped to float values, or an 'Error' key if detection failed.
    """
    result = {}

    # --- Always: detect faces ---
    faces = detector.detect_faces(img_array)
    if not faces or not faces[0]:
        return {"Error": "No face detected"}

    # --- Always: detect landmarks (needed for metrics) ---
    landmarks = detector.detect_landmarks(img_array, detected_faces=faces)
    if not landmarks or not landmarks[0]:
        return {"Error": "Landmark detection failed"}

    # Take the first detected face
    lm = np.array(landmarks[0][0])  # shape (68, 2)

    # Store landmark coordinates
    result["_Raw_Landmarks"] = lm.tolist()
    landmark_array_to_result(lm, result)

    # Always: compute derived metrics from landmarks (fast - pure math)
    try:
        result["Eyebrow_V"] = round(calculate_eyebrow_v_shape(result), 4)
        result["fWHR"] = round(calculate_fwhr(result), 4)
    except Exception:
        result["Eyebrow_V"] = None
        result["fWHR"] = None

    # --- Optional: Action Units (HOG extraction + XGB - slowest step) ---
    if detect_aus:
        try:
            aus = detector.detect_aus(img_array, landmarks)
            if aus is not None and len(aus) > 0:
                au_frame = np.array(aus[0])
                au_vals = au_frame[0] if au_frame.ndim == 2 else au_frame
                for col, val in zip(AU_COLUMNS, au_vals):
                    result[col] = round(float(val), 4)
        except Exception:
            pass  # AUs unavailable for this image

    # --- Optional: Emotions ---
    if detect_emotions:
        try:
            emotions = detector.detect_emotions(img_array, faces, landmarks)
            if emotions is not None and len(emotions) > 0:
                emo_frame = np.array(emotions[0])
                emo_vals = emo_frame[0] if emo_frame.ndim == 2 else emo_frame
                for col, val in zip(EMOTION_COLUMNS, emo_vals):
                    result[col] = round(float(val), 4)
        except Exception:
            pass  # Emotions unavailable for this image

    # --- Optional: Head Pose (img2pose - runs its own face detection internally) ---
    if detect_pose:
        try:
            poses_dict = detector.detect_facepose(img_array, landmarks)
            poses = poses_dict.get("poses", [])
            if poses and poses[0]:
                p = np.array(poses[0][0])
                result["Pitch"] = round(float(p[0]), 4)
                result["Roll"] = round(float(p[1]), 4)
                result["Yaw"] = round(float(p[2]), 4)
        except Exception:
            pass  # Pose unavailable for this image

    return result


# ---------------------------------------------------------------------------
# Landmark visualisation
# ---------------------------------------------------------------------------

def draw_landmarks_on_image(image: Image.Image, landmarks_data: dict) -> Image.Image:
    """Overlay 68-point facial landmarks and connection lines on an image.

    Dot and line sizes are scaled proportionally to image dimensions so
    the visualisation looks reasonable on both small thumbnails and
    high-resolution photographs.

    Args:
        image:          A PIL Image (RGB).
        landmarks_data: dict mapping 'LM_{i}_X' / 'LM_{i}_Y' to pixel
                        coordinates.

    Returns:
        PIL.Image.Image with landmarks drawn on top.
    """
    img_with_landmarks = image.copy()
    draw = ImageDraw.Draw(img_with_landmarks)
    width, height = img_with_landmarks.size
    dot_radius = max(2, int(min(width, height) * 0.008))
    line_width = max(1, dot_radius // 2)

    # Build lookup of (x, y) per landmark index once
    pts = {}
    for i in range(68):
        x = landmarks_data.get(f"LM_{i}_X")
        y = landmarks_data.get(f"LM_{i}_Y")
        if x is not None and y is not None:
            pts[i] = (x, y)
            draw.ellipse(
                [(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)],
                fill="yellow", outline="yellow",
            )

    # Draw lines for facial feature groups
    for g_idx, group in enumerate(_LANDMARK_GROUPS):
        for j in range(len(group) - 1):
            p1, p2 = pts.get(group[j]), pts.get(group[j + 1])
            if p1 and p2:
                draw.line([p1, p2], fill="lime", width=line_width)
        # Close loops for eyes, lips
        if g_idx in _CLOSED_GROUPS:
            p1, p2 = pts.get(group[0]), pts.get(group[-1])
            if p1 and p2:
                draw.line([p1, p2], fill="lime", width=line_width)

    return img_with_landmarks


# ---------------------------------------------------------------------------
# Output explanation dialog
# ---------------------------------------------------------------------------

def render_output_explainer():
    """Render a short guide to the landmark output."""
    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[role="dialog"] {
            width: min(96vw, 1280px);
            max-width: min(96vw, 1280px);
        }

        div[data-testid="stDialog"] div[role="dialog"] [data-testid="stVerticalBlock"] {
            gap: 0.45rem;
        }

        div[data-testid="stDialog"] [data-testid="stImage"] img {
            max-height: 72vh;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "FaceMeasure identifies 68 standard facial landmarks (using "
        "MobileFaceNet) for each detected face (identified using RetinaFace), "
        "and returns the X- and Y-coordinate for each landmark."
    )

    original_col, landmarks_col = st.columns(2)
    with original_col:
        st.image("static/1_original.png", caption="Original image", use_container_width=True)
    with landmarks_col:
        st.image("static/5_landmarksClose.jpeg", caption="Facial landmarks", use_container_width=True)


def render_metrics_explainer():
    """Render a short guide to additional derived facial metrics."""
    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[role="dialog"] {
            width: min(96vw, 1280px);
            max-width: min(96vw, 1280px);
        }

        div[data-testid="stDialog"] div[role="dialog"] [data-testid="stVerticalBlock"] {
            gap: 0.45rem;
        }

        div[data-testid="stDialog"] [data-testid="stImage"] img {
            max-height: 72vh;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Eyebrow V-shape")
    st.markdown("*(Witkower, Khanna, & Rule, in prep)*")
    st.markdown(
        "Eyebrow V-shape is calculated from the slopes of the inner eyebrow "
        "landmarks. FaceMeasure fits a linear model to the the standardized "
        "landmark coordinates for each eyebrow, reverse-codes the right "
        "eyebrow slope, and averages the two slopes into a single V-shape "
        "value. Higher positive values indicate a stronger V-shape pattern."
    )

    st.image("static/Vshape.png", caption="Eyebrow V-shape", use_container_width=True)


dialog_decorator = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)


def output_dialog(title: str):
    """Return a dialog decorator, using a wider modal when Streamlit supports it."""
    try:
        if "width" in inspect.signature(dialog_decorator).parameters:
            return dialog_decorator(title, width="large")
    except (TypeError, ValueError):
        pass
    return dialog_decorator(title)

if dialog_decorator:
    @output_dialog("Understanding the output")
    def show_output_explainer():
        render_output_explainer()

    @output_dialog("Additional facial metrics")
    def show_metrics_explainer():
        render_metrics_explainer()
else:
    def show_output_explainer():
        st.session_state["show_output_explainer_fallback"] = True

    def show_metrics_explainer():
        st.session_state["show_metrics_explainer_fallback"] = True


# ---------------------------------------------------------------------------
# Upload form
# ---------------------------------------------------------------------------
with st.form("upload_form", clear_on_submit=True, border=False):
    uploaded_images = st.file_uploader(label = "Upload file(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="hidden")
    submitted = st.form_submit_button("Analyze image(s)", width='stretch', type="primary")

# ---------------------------------------------------------------------------
# Analysis options — toggles in the main body (not the sidebar)
# Researchers enable only the analyses they need; everything else is skipped
# at inference time, which dramatically reduces per-image processing time.
# ---------------------------------------------------------------------------
with st.expander("Analysis options", icon=":material/tune:", expanded=False):
    st.markdown("**Landmark options**")
    landmark_col1, landmark_col2 = st.columns(2)
    with landmark_col1:
        st.toggle(
            "Facial Landmarking",
            value=True,
            disabled=True,
            help="This is a fundamental feature of facemeasure.",
        )
    with landmark_col2:
        use_procrustes_rotation_flag = st.toggle(
            "Use Procrustes rotation (standardizes landmarks to eliminate effects of positioning and scale)",
            value=True,
            help="Center, scale, and rotate landmark coordinates to a common Procrustes shape for cross-target comparisons.",
        )

    st.markdown("**Additional output options**")
    output_col1, output_col2, output_col3 = st.columns(3)
    with output_col1:
        detect_aus_flag = st.toggle(
            "Action Units",
            value=False,
            help="Detect 20 facial action units (AU01–AU43). Adds ~2–5 s per image.",
        )
    with output_col2:
        detect_emotions_flag = st.toggle(
            "Emotions",
            value=False,
            help="Estimate intensity of basic emotion expressions (anger, disgust, fear, happiness, sadness, surprise, neutral).",
        )
    with output_col3:
        detect_pose_flag = st.toggle(
            "Head Pose",
            value=False,
            help="Estimate head orientation (pitch, roll, yaw). Adds ~1–2 s per image.",
        )
    st.caption("Landmarks, fWHR, and eyebrow V-shape are always computed. Procrustes standardization is used by default for comparable landmark coordinates. Each additional feature adds processing time per image.")

st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) button {
        min-height: 2rem;
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
        font-size: 0.85rem;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

explainer_col1, explainer_col2 = st.columns(2)
with explainer_col1:
    if st.button("Landmark explanation", type="secondary", width='stretch'):
        show_output_explainer()
with explainer_col2:
    if st.button("Additional facial metrics", type="secondary", width='stretch'):
        show_metrics_explainer()

if not dialog_decorator and st.session_state.get("show_output_explainer_fallback"):
    with st.container(border=True):
        render_output_explainer()

if not dialog_decorator and st.session_state.get("show_metrics_explainer_fallback"):
    with st.container(border=True):
        render_metrics_explainer()

if submitted:
    if not uploaded_images:
        st.warning("⚠️ Please upload at least one image before analyzing.")
    else:
        results = []
        progress_bar = st.progress(0, "Analyzing...")
        total_images = len(uploaded_images)
        start_time = time.time()

        try:
            detector = load_detector()

            for idx, img_file in enumerate(uploaded_images):
                try:
                    img_file.seek(0)
                    pil_img = Image.open(img_file).convert("RGB")
                    img_array = np.array(pil_img)

                    with torch.no_grad():
                        data = analyze_image(
                            detector, img_array,
                            detect_aus=detect_aus_flag,
                            detect_emotions=detect_emotions_flag,
                            detect_pose=detect_pose_flag,
                        )
                except Exception as img_err:
                    data = {"Error": str(img_err)}
                data["Image_Name"] = img_file.name
                results.append(data)
                progress_bar.progress((idx + 1) / total_images)

            if use_procrustes_rotation_flag:
                apply_procrustes_to_results(results)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            results = [{"Error": str(e), "Image_Name": img.name} for img in uploaded_images]

        elapsed = time.time() - start_time
        progress_bar.empty()
        export_results = [
            {k: v for k, v in row.items() if not k.startswith("_")}
            for row in results
        ]
        df = pd.DataFrame(export_results)

        # Reorder columns: most useful first, raw landmarks last
        present = set(df.columns)
        metric_cols = [c for c in ("fWHR", "Eyebrow_V") if c in present]
        au_cols = [c for c in AU_COLUMNS if c in present]
        emotion_cols = [c for c in EMOTION_COLUMNS if c in present]
        pose_cols = [c for c in ("Pitch", "Roll", "Yaw") if c in present]
        landmark_cols = [c for c in df.columns if c.startswith("LM_")]
        ordered = ["Image_Name"] + metric_cols + au_cols + emotion_cols + pose_cols + landmark_cols
        ordered_set = set(ordered)
        other_cols = [c for c in df.columns if c not in ordered_set]
        df = df[[c for c in ordered + other_cols if c in present]]

        # Summary of features computed
        feat_list = ["Landmarks", "fWHR", "Eyebrow V"]
        if detect_aus_flag:
            feat_list.append("Action Units")
        if detect_emotions_flag:
            feat_list.append("Emotions")
        if detect_pose_flag:
            feat_list.append("Head Pose")
        if use_procrustes_rotation_flag:
            feat_list.append("Procrustes-aligned landmarks")

        # Display results
        st.write("&nbsp;")
        st.subheader("Analysis Results")
        st.caption(f"Analyzed {total_images} image(s) in {elapsed:.1f}s  ·  Features: {', '.join(feat_list)}")
        
        with st.container(border=True):
            st.dataframe(df, hide_index=True)

            # Generate file name with current date and time
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            csv_filename = f"facemeasure_{timestamp}.csv"

            col_a, col_b = st.columns(2)
            
            # Provide download button for CSV
            csv = df.to_csv(index=False)
            col_a.download_button(
                "Download results as CSV", 
                data=csv, 
                file_name=csv_filename, 
                mime="text/csv", 
                width='stretch', 
                type="primary")  

            # Provide download button for JSON
            json_data = json.dumps(export_results, indent=2)
            json_filename = f"facemeasure_{timestamp}.json"
            col_b.download_button(
                "Download results as JSON",
                data=json_data,
                file_name=json_filename,
                mime="application/json",    
                width='stretch',
                type="primary"
            )

        # === Show landmarks on a randomly chosen image ===
        with st.container(border=True):
            valid_results = [r for r in results if "Error" not in r]
            if valid_results:
                # Pick a random index
                random_idx = random.randrange(len(valid_results))
                chosen_result = valid_results[random_idx]
                chosen_img_name = chosen_result["Image_Name"]
                
                st.write(f"Visualizing landmarks on a randomly chosen image (`{chosen_img_name}`) from your uploads:")
                
                for img in uploaded_images:
                    if img.name == chosen_img_name:
                        img.seek(0)
                        with Image.open(img) as pil_img:
                            # Draw landmarks on grayscale image
                            pil_gray = ImageOps.grayscale(pil_img)
                            pil_gray_rgb = pil_gray.convert("RGB")  # So landmarks appear in color
                            raw_landmarks = chosen_result.get("_Raw_Landmarks")
                            if raw_landmarks:
                                landmark_overlay = {}
                                for i, (x, y) in enumerate(raw_landmarks):
                                    landmark_overlay[f"LM_{i}_X"] = x
                                    landmark_overlay[f"LM_{i}_Y"] = y
                            else:
                                landmark_overlay = chosen_result
                            img_with_landmarks = draw_landmarks_on_image(pil_gray_rgb, landmark_overlay)
                            
                            # Show side by side using Streamlit columns
                            col_c, col_d = st.columns(2)
                            with col_c:
                                st.image(pil_img, caption="Original", use_container_width=True)
                            with col_d:
                                st.image(img_with_landmarks, caption="With facial landmarks", use_container_width=True)
                        break
            else:
                st.info("No valid faces detected in uploaded images, so cannot show landmarks.")
