import streamlit as st
import dlib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from streamlit_lottie import st_lottie
from datetime import datetime

# Set up the Streamlit page configuration
st.set_page_config(page_icon="🧑", page_title="FaceMeasure", layout="centered")

# Hide Streamlit default menu, footer, and header
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Main page content
st.title("🧑 FaceMeasure")
st.write("_Democratizing objective face measurement_")

# Sidebar information
st.sidebar.title("🧑 FaceMeasure")
st.sidebar.write("_Democratizing objective face measurement_")
st.sidebar.info("🌱 Supported by the Social and Behavioural Data Science Centre, University of Amsterdam")
st.sidebar.info("🙈 No identifiable data is stored. Refresh to clear data.")
st.sidebar.info("🛠️ Open source: [GitHub](https://github.com/saurabh-khanna/facemeasure)")

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

col1, col2 = st.columns([1.75, 1])

with col2:
    st_lottie("https://lottie.host/a0357f2b-b951-4f69-b9e8-35b36e79b386/9B0IQaQ77q.json")

# Allow multiple image uploads using a Streamlit form
with col1:
    st.write("FaceMeasure democratizes facial analysis by allowing researchers to obtain precise facial metrics instantly without costly software or coding.")
    
    with st.form("upload_form", clear_on_submit=True, border=False):
        uploaded_images = st.file_uploader("Upload one or more image(s) to get started.", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
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
        st.subheader("Analysis Results")
        st.dataframe(df, hide_index=True)

        # Generate file name with current date and time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = f"facemeasure_{timestamp}.csv"

        # Provide download button for CSV
        csv = df.to_csv(index=False)
        st.download_button("Download results as CSV", data=csv, file_name=csv_filename, mime="text/csv")
