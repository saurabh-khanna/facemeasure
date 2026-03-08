# 👤 facemeasure

[![Streamlit App](https://img.shields.io/badge/Streamlit-Online-brightgreen)](https://facemeasure.com)
[![License](https://img.shields.io/github/license/saurabh-khanna/facemeasure)](LICENSE)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/saurabh-khanna/facemeasure)

**facemeasure** democratizes facial landmark analysis by allowing anyone to extract precise facial metrics from images - no expensive software or coding skills needed. Upload facial images, visualize facial landmarks, and download results as CSV or JSON. Everything runs in your browser, and no personal data is stored.

## 🚀 Features

* **Facial Landmark Detection**: Instantly extract 68 facial landmarks from each face using py-feat's modular detection pipeline.
* **Derived Metrics**: Automatically computes facial width-to-height ratio (fWHR) and eyebrow V-shape index from detected landmarks.
* **Action Unit Detection**: Optionally detect 20 facial action units (AU01–AU43) using HOG features and XGBoost classifiers.
* **Emotion Classification**: Optionally classify 7 basic emotions (anger, disgust, fear, happiness, sadness, surprise, neutral).
* **Head Pose Estimation**: Optionally estimate head orientation (pitch, roll, yaw) via img2pose.
* **Batch Processing**: Upload and analyze multiple images at once.
* **Interactive Visualization**: See landmarks drawn and connected over each face.
* **Download Results**: Export all results as CSV or JSON.
* **Privacy-Friendly**: No images or data are ever stored—refresh the app to clear everything.
* **Free & Open Source**: Supported by the Social and Behavioural Data Science Centre, University of Amsterdam.


## 🖥️ Demo

Try it live:
[👉 facemeasure](https://facemeasure.com)

## 📦 Installation

### Option 1: Use the Live App (No Installation)

Visit [facemeasure.com](https://facemeasure.com) — nothing to install!

### Option 2: Run Locally with Python

**Requirements**: Python 3.9 or later

```bash
# Clone the repository
git clone https://github.com/saurabh-khanna/facemeasure.git
cd facemeasure

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run home.py
```

The app will open in your browser at `http://localhost:8501`.

### Option 3: Run with Docker

**Requirements**: Docker installed on your system

```bash
# Clone the repository
git clone https://github.com/saurabh-khanna/facemeasure.git
cd facemeasure

# Build the Docker image
docker build -t facemeasure .

# Run the container
docker run -p 8501:8501 facemeasure
```

Access the app at `http://localhost:8501`.

### Running Tests

```bash
pip install pytest
pytest test_pipeline.py -v
```

## 🧑‍🔬 How to Use

1. **Upload** one or more JPEG or PNG images containing faces.
2. **Click “Analyze image(s)”** to extract and display facial landmarks.
3. **Download results** as a CSV or JSON file.
4. **See a random visualization** of detected facial landmarks over one of your uploaded images.


## 📝 Output

* **CSV**: All metrics and landmarks for each image (columns: `Image_Name`, `fWHR`, `Eyebrow_V`, `AU01`–`AU43`, emotions, `Pitch`, `Roll`, `Yaw`, `LM_0_X`, `LM_0_Y`, ...).
* **JSON**: Structured array of objects with the same keys.
* **Visualization**: Original and landmark-overlaid images shown side by side.


## 📚 Dependencies

* [Streamlit](https://streamlit.io/)
* [py-feat](https://py-feat.org/)
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [Pillow](https://python-pillow.org/)


## 🙏 Acknowledgments

* Supported by the Social and Behavioural Data Science Centre, University of Amsterdam.

## 🛡️ License

This project is [AGPLv3 licensed](LICENSE).

## 💡 Contributing

Contributions are welcome!
Feel free to open issues, create pull requests, or suggest features.
