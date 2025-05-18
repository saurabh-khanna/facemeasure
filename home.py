import streamlit as st
import dlib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw # Added ImageDraw
from streamlit_lottie import st_lottie
from datetime import datetime
import random
import json

# Set up the Streamlit page configuration
st.set_page_config(page_icon="👤", page_title="FaceMeasure", layout="centered")

# Hide Streamlit default menu, footer, and header
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stElementToolbar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main page content
st.title(":bust_in_silhouette: facemeasure")

col1, col2, col3 = st.columns([1, 1.618, 1])

with col2:
    st_lottie("https://lottie.host/71c80b64-c8c4-41a8-a469-ad6ba3555abe/RK6dp4pBsY.json")

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
        <b>facemeasure</b> democratizes facial analysis by allowing researchers to obtain precise facial metrics instantly without relying on expensive software or programming skills. Upload one or more facial image(s) to get started<b><span class="blinking-underscore">_</span></b>
    </p>
    """, unsafe_allow_html=True) 

# Sidebar information
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

# Load Dlib models
@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file exists
    return detector, predictor

def extract_landmarks(image):
    """Extract 68 facial landmarks from the image using Dlib."""
    detector, predictor = load_models()
    
    img_gray = ImageOps.grayscale(image)  # Convert to grayscale
    img_array = np.array(img_gray)  # Convert PIL image to NumPy array
    faces = detector(img_array)

    if not faces:
        return {"Error": "No face detected"}

    landmarks = predictor(img_array, faces[0])
    landmark_dict = {}
    for i in range(68):
        landmark_dict[f"LM_{i}_X"] = landmarks.part(i).x
        landmark_dict[f"LM_{i}_Y"] = landmarks.part(i).y
    return landmark_dict


def draw_landmarks_on_image(image: Image.Image, landmarks_data: dict) -> Image.Image:
    """Draws landmarks with size scaled to image, and connects groups correctly."""
    img_with_landmarks = image.copy()
    draw = ImageDraw.Draw(img_with_landmarks)
    width, height = img_with_landmarks.size
    dot_radius = max(2, int(min(width, height) * 0.008))

    # Draw points
    for i in range(68):
        x = landmarks_data.get(f"LM_{i}_X")
        y = landmarks_data.get(f"LM_{i}_Y")
        if x is not None and y is not None:
            draw.ellipse([(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)], fill="yellow", outline="yellow")

    # Draw lines for facial features
    landmark_connections = [
        list(range(0, 17)),      # Jawline
        list(range(17, 22)),     # Left eyebrow
        list(range(22, 27)),     # Right eyebrow
        list(range(27, 31)),     # Nose bridge
        list(range(31, 36)),     # Lower nose
        list(range(36, 42)),     # Left eye
        list(range(42, 48)),     # Right eye
        list(range(48, 60)),     # Outer lip
        list(range(60, 68)),     # Inner lip
    ]

    for group in landmark_connections:
        for i in range(len(group) - 1):
            x1, y1 = landmarks_data.get(f"LM_{group[i]}_X"), landmarks_data.get(f"LM_{group[i]}_Y")
            x2, y2 = landmarks_data.get(f"LM_{group[i+1]}_X"), landmarks_data.get(f"LM_{group[i+1]}_Y")
            if None not in (x1, y1, x2, y2):
                draw.line([(x1, y1), (x2, y2)], fill="lime", width=max(1, dot_radius//2))
        # For closed curves, connect last to first
        if group in [list(range(36, 42)), list(range(42, 48)), list(range(48, 60)), list(range(60, 68))]:
            x1, y1 = landmarks_data.get(f"LM_{group[0]}_X"), landmarks_data.get(f"LM_{group[0]}_Y")
            x2, y2 = landmarks_data.get(f"LM_{group[-1]}_X"), landmarks_data.get(f"LM_{group[-1]}_Y")
            if None not in (x1, y1, x2, y2):
                draw.line([(x1, y1), (x2, y2)], fill="lime", width=max(1, dot_radius//2))

    return img_with_landmarks


# Allow multiple image uploads using a Streamlit form
with st.form("upload_form", clear_on_submit=True, border=False):
    uploaded_images = st.file_uploader(label = "Upload file(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="hidden")
    submitted = st.form_submit_button("Analyze image(s)", use_container_width=True, type="primary")

if submitted:
    if not uploaded_images:
        st.warning("⚠️ Please upload at least one image before analyzing.")
    else:
        results = []
        progress_bar = st.progress(0, "Analyzing...")
        total_images = len(uploaded_images)
        
        for idx, img in enumerate(uploaded_images):
            with Image.open(img) as image:
                data = extract_landmarks(image)
                data["Image_Name"] = img.name  # Add image name
                results.append(data)
            progress_bar.progress((idx + 1) / total_images)
        
        progress_bar.empty()
        df = pd.DataFrame(results)
        df = df[["Image_Name"] + [col for col in df.columns if col != "Image_Name"]]  # Ensure image name is the first column

        # Display results
        st.write("&nbsp;")
        st.subheader("Analysis Results")
        
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
                use_container_width=True, 
                type="primary")  

            # Provide download button for JSON
            json_data = json.dumps(results, indent=2)
            json_filename = f"facemeasure_{timestamp}.json"
            col_b.download_button(
                "Download results as JSON",
                data=json_data,
                file_name=json_filename,
                mime="application/json",    
                use_container_width=True,
                type="primary"
            )

        # === NEW: Show landmarks on a randomly chosen image ===
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
