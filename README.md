# Chest-X-ray-Classifier
AI-based chest X-ray classifier for detecting COVID-19, Pneumonia, and Normal conditions.

# PulmoCheck AI 🫁

PulmoCheck AI is an advanced, AI-powered diagnostic dashboard designed to analyze chest X-ray images and detect COVID-19, Pneumonia, or Normal lung conditions. Built with a Convolutional Neural Network (CNN) and deployed via Streamlit, the application features a modern, premium glassmorphism interface tailored for clinical review.

## Features ✨

* **AI-Powered Diagnostics**: Utilizes a trained deep learning CNN model to classify X-rays into three categories: COVID-19, Pneumonia, or Normal.
* **Premium Dashboard**: A sleek, dark-themed UI with glassmorphism effects, custom typography, and responsive layouts for a professional user experience.
* **Demo Samples**: Quick-load buttons for COVID-19, Normal, and Pneumonia scans located at the bottom left for rapid demonstration and testing.
* **Scan Validation**: Automatically analyzes image orientation (AP/PA views) and inspiration quality (rib count detection) using OpenCV.
* **Region of Interest (ROI) Analysis**: Provides insights into primary focus areas, density profiles, and symmetry variances based on the AI's classification.
* **Confidence Spectrum**: Visualizes the model's prediction confidence across all three classes with dynamic progress bars.
* **Diagnostic Notes**: Generates automated analysis notes consistent with the detected radiological patterns.
* **Advanced Image Toolset**: Functional viewer toolbar for real-time image manipulation:
    * 🔍 **Invert Colors**: Toggle negative view for identifying subtle dense structures.
    * 🫁 **Density Heatmap**: Pseudo-color mapping to visualize lung density variations.
    * ⚡ **Contrast Enhancement**: CLAHE-based optimization for balanced and sharper lung field visualization.

## Technologies Used 🛠️

* **Python 3.x**
* **Streamlit**: For the interactive web dashboard.
* **TensorFlow / Keras**: Deep learning framework for the CNN model.
* **OpenCV (cv2)**: For image processing and anatomical analysis.
* **NumPy & Pillow (PIL)**: For array manipulations and image handling.

## Disclaimer ⚠️

**AI-GENERATED SUGGESTION — FOR CLINICAL REVIEW ONLY. FINAL DIAGNOSIS IS A MEDICAL DECISION.**
This tool is intended for educational and research purposes. It is not a substitute for professional medical advice, diagnosis, or treatment.
