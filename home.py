import streamlit as st

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

# main page
st.title("🧑 FaceMeasure")
st.write("_Democratizing objective face measurement_")
st.write("&nbsp;")

# sidebar
st.sidebar.title("🧑 FaceMeasure")
st.sidebar.write("_Democratizing objective face measurement_")
st.sidebar.write("&nbsp;")
st.sidebar.info("🌱 Supported by the Social and Behavioural Data Science Centre, University of Amsterdam")
st.sidebar.info("🙈 We do not collect any identifiable information. All data is cleared once you close this web page.")
st.sidebar.info("🛠️ Our code base is fully open source and can be found [here]().")

st.write("FaceMeasure democratizes facial analysis by allowing researchers to obtain precise facial metrics instantly without costly software or coding. Get started by uploading a picture, or take one using your webcam.")

# Create a sidebar for user options
option = st.radio(label = "invisible", options = ("Upload a picture", "Take a picture"), label_visibility = "collapsed")

if option == "Upload a picture":
    image = st.file_uploader(label = "invisible", type=["jpg", "jpeg", "png"], label_visibility = "collapsed")
    if image is not None:
        st.image(image, caption='Uploaded Image')
elif option == "Take a picture":
    image = st.camera_input(label = "invisible", label_visibility = "collapsed")