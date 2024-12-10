import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import pandas as pd
from streamlit_lottie import st_lottie

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
st.write("&nbsp;")

# Sidebar information
st.sidebar.title("🧑 FaceMeasure")
st.sidebar.write("_Democratizing objective face measurement_")
st.sidebar.write("&nbsp;")
st.sidebar.info("🌱 Supported by the Social and Behavioural Data Science Centre, University of Amsterdam")
st.sidebar.info("🙈 We do not collect any identifiable information. All data is cleared once you refresh or close this web page.")
st.sidebar.info("🛠️ Our code base is fully open source and can be found [here]()")


def load_image(image_file):
    """Load an image from a file."""
    return Image.open(image_file)


def analyze_image(image):
    """Analyze an image and return its attributes."""
    img_rgb = np.array(image.convert("RGB"))
    analysis = DeepFace.analyze(
        img_path=img_rgb,
        actions=["age", "gender", "race", "emotion"]
    )
    result = {
        "Age": analysis[0]["age"],
        "Dominant Gender": analysis[0]["dominant_gender"],
        "Dominant Race": analysis[0]["dominant_race"],
        "Dominant Emotion": analysis[0]["dominant_emotion"],
        "Face Confidence (%)": f"{analysis[0]['face_confidence'] * 100:.1f}"
    }
    return result


col1, col2 = st.columns([2.4, 1])

with col2:
    st_lottie("https://lottie.host/a0357f2b-b951-4f69-b9e8-35b36e79b386/9B0IQaQ77q.json")

# Allow multiple image uploads

with col1:
    st.write("FaceMeasure democratizes facial analysis by allowing researchers to obtain precise facial metrics instantly without costly software or coding. Upload multiple pictures to get started.")
    
    uploaded_images = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        results = []

        # Process each uploaded image
        for image_file in uploaded_images:
            image = load_image(image_file)
            result = analyze_image(image)
            results.append(result)

        # Convert results to a DataFrame
        df = pd.DataFrame(results)

        # Display the DataFrame
        st.subheader("Analysis Results")
        st.dataframe(df)

        # Provide a button to download the DataFrame as a CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="facial_analysis_results.csv",
            mime="text/csv"
        )
