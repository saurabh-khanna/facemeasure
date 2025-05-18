# 👤 facemeasure

[![Streamlit App](https://img.shields.io/badge/Streamlit-Online-brightgreen)](https://facemeasure.applikuapp.com)
[![License](https://img.shields.io/github/license/saurabh-khanna/facemeasure)](LICENSE)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/saurabh-khanna/facemeasure)

**facemeasure** democratizes facial landmark analysis by allowing anyone to extract precise facial metrics from images—no expensive software or coding skills needed. Upload facial images, visualize facial landmarks, and download results as CSV or JSON. Everything runs in your browser, and no personal data is stored.

## 🚀 Features

* **Facial Landmark Detection**: Instantly extract 68 facial landmarks from each face in your images using dlib’s shape predictor.
* **Batch Processing**: Upload and analyze multiple images at once.
* **Interactive Visualization**: See landmarks drawn and connected over each face (on grayscale or original images).
* **Download Results**: Export all detected landmarks as CSV or JSON.
* **Privacy-Friendly**: No images or data are ever stored—refresh the app to clear everything.
* **Free & Open Source**: Supported by the Social and Behavioural Data Science Centre, University of Amsterdam.


## 🖥️ Demo

Try it live:
[👉 facemeasure](https://facemeasure.applikuapp.com)

## 🧑‍🔬 How to Use

1. **Upload** one or more JPEG or PNG images containing faces.
2. **Click “Analyze image(s)”** to extract and display facial landmarks.
3. **Download results** as a CSV or JSON file.
4. **See a random visualization** of detected facial landmarks over one of your uploaded images.


## 📝 Output

* **CSV**: All landmarks for each image (columns: `Image_Name`, `LM_0_X`, `LM_0_Y`, ...).
* **JSON**: Structured array of objects with the same landmark keys.
* **Visualization**: Original and landmark-overlaid images shown side by side.


## 📚 Dependencies

* [Streamlit](https://streamlit.io/)
* [dlib](https://github.com/davisking/dlib)
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
