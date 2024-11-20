import streamlit as st
from deepface import DeepFace
import numpy as np
from PIL import Image
import pandas as pd

# Set up the Streamlit page configuration and hide menu, footer, header
st.set_page_config(page_icon="🧑", page_title="FaceMeasure", layout="centered")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Main page
st.title("🧑 FaceMeasure")
st.write("_Democratizing objective face measurement_")
st.write("&nbsp;")

# Sidebar
st.sidebar.title("🧑 FaceMeasure")
st.sidebar.write("_Democratizing objective face measurement_")
st.sidebar.write("&nbsp;")
st.sidebar.info("🌱 Supported by the Social and Behavioural Data Science Centre, University of Amsterdam")
st.sidebar.info("🙈 We do not collect any identifiable information. All data is cleared once you refresh or close this web page.")
st.sidebar.info("🛠️ Our code base is fully open source and can be found [here](https://github.com/saurabh-khanna/facemeasure/).")

st.write("FaceMeasure democratizes facial analysis by allowing researchers to obtain precise facial metrics instantly without costly software or coding. Get started by uploading a picture, or take one using your webcam.")

# Create a sidebar for user options
option = st.radio(label="invisible", options=("Upload a picture", "Take a picture"), label_visibility="collapsed")

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

# Image handling based on user option
if option == "Upload a picture":
    image = st.file_uploader(label="invisible", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if image is not None:
        img = load_image(image)
        st.image(img, caption='Uploaded picture')

elif option == "Take a picture":
    image = st.camera_input(label="invisible", label_visibility="collapsed")
    if image is not None:
        img = load_image(image)

st.write("&nbsp;")
if image is not None:
    with st.status("Processing your picture..."):
        # Convert RGBA to RGB if needed
        if img.shape[-1] == 4:
            img = Image.fromarray(img).convert("RGB")
            img = np.array(img)

        # Perform analysis
        objs = DeepFace.analyze(
            img_path=img,  # Pass the numpy array here
            actions=['age', 'gender', 'race', 'emotion'],
        )

    # Extract the attributes from the 'objs' list
    attributes = {
        'Age': objs[0]['age'],
        'Dominant Gender': objs[0]['dominant_gender'],
        'Dominant Race': objs[0]['dominant_race'],
        'Dominant Emotion': objs[0]['dominant_emotion'],
        'Face Confidence': f"{objs[0]['face_confidence'] * 100:.1f}%",
    }

    # Round percentages in the detailed attributes
    gender_df = pd.DataFrame(objs[0]['gender'], index=[0]).applymap(lambda x: f"{x:.1f}%")
    race_df = pd.DataFrame(objs[0]['race'], index=[0]).applymap(lambda x: f"{x:.1f}%")
    emotion_df = pd.DataFrame(objs[0]['emotion'], index=[0]).applymap(lambda x: f"{x:.1f}%")

    # Access region details including eye coordinates
    region = objs[0]['region']
    region_df = pd.DataFrame([{
        'x': region.get('x'),
        'y': region.get('y'),
        'w': region.get('w'),
        'h': region.get('h'),
        'left_eye_x': region.get('left_eye', [None, None])[0],
        'left_eye_y': region.get('left_eye', [None, None])[1],
        'right_eye_x': region.get('right_eye', [None, None])[0],
        'right_eye_y': region.get('right_eye', [None, None])[1]
    }])

    # Display the attributes
    st.subheader("Key Facial Attributes")
    st.table(pd.DataFrame(attributes.items(), columns=['Attribute', 'Value']).reset_index(drop=True))

    # Display region details including eyes
    st.subheader("Face Region and Eye Coordinates")
    st.table(region_df.reset_index(drop=True))

    # Display gender probabilities
    st.subheader("Gender Confidence Levels")
    st.table(gender_df.reset_index(drop=True))

    # Display race probabilities
    st.subheader("Race Confidence Levels")
    st.table(race_df.reset_index(drop=True))

    # Display emotion probabilities
    st.subheader("Emotion Confidence Levels")
    st.table(emotion_df.reset_index(drop=True))
