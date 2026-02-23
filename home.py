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


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_lottie_url(url: str):
    """Fetch a Lottie animation JSON from *url*, cached across reruns."""
    r = requests.get(url)
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
        st_lottie(lottie_json)

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
    </style>
    <p style='text-align: left; font-size: 18px;'>
        <b>facemeasure</b> democratizes facial analysis by allowing researchers
        to obtain precise facial metrics instantly without relying on expensive
        software or programming skills. Upload one or more facial image(s) to
        get started<b><span class="blinking-underscore">_</span></b>
    </p>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Analysis options — toggles in the main body (not the sidebar)
# Researchers enable only the analyses they need; everything else is skipped
# at inference time, which dramatically reduces per-image processing time.
# ---------------------------------------------------------------------------
with st.expander("Analysis options", icon=":material/tune:"):
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        detect_aus_flag = st.toggle(
            "Action Units (AUs)",
            value=False,
            help="Detect 20 facial action units (AU01–AU43). Adds ~2–5 s per image.",
        )
    with opt_col2:
        detect_emotions_flag = st.toggle(
            "Emotions",
            value=False,
            help="Detect 7 basic emotions (anger, disgust, fear, happiness, sadness, surprise, neutral).",
        )
    with opt_col3:
        detect_pose_flag = st.toggle(
            "Head Pose",
            value=False,
            help="Estimate head orientation (pitch, roll, yaw). Adds ~1–2 s per image.",
        )
    st.caption("Landmarks, fWHR, and eyebrow V-shape are always computed. Each additional feature adds processing time per image.")

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
    appearance (Hehman et al., 2015).  Landmark coordinates are first
    z-standardised across all 68 points so the result is scale-invariant.

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
    for i in range(68):
        result[f"LM_{i}_X"] = round(float(lm[i, 0]), 4)
        result[f"LM_{i}_Y"] = round(float(lm[i, 1]), 4)

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
# Upload form
# ---------------------------------------------------------------------------
with st.form("upload_form", clear_on_submit=True, border=False):
    uploaded_images = st.file_uploader(label = "Upload file(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="hidden")
    submitted = st.form_submit_button("Analyze image(s)", width='stretch', type="primary")

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

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            results = [{"Error": str(e), "Image_Name": img.name} for img in uploaded_images]

        elapsed = time.time() - start_time
        progress_bar.empty()
        df = pd.DataFrame(results)

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
            json_data = json.dumps(results, indent=2)
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
                            img_with_landmarks = draw_landmarks_on_image(pil_gray_rgb, chosen_result)
                            
                            # Show side by side using Streamlit columns
                            col_c, col_d = st.columns(2)
                            with col_c:
                                st.image(pil_img, caption="Original", use_container_width=True)
                            with col_d:
                                st.image(img_with_landmarks, caption="With facial landmarks", use_container_width=True)
                        break
            else:
                st.info("No valid faces detected in uploaded images, so cannot show landmarks.")
