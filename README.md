# Video Meeting Engagement Analyzer

An academic research and portfolio project for estimating participant engagement in recorded video meetings. 

Developed as a demonstration of Computer Vision, Deep Learning, and Classical Machine Learning pipelines for video-conference analysis.

---

## 🛠️ Project Structure

```text
online meating analyser/
│
├── data/                      # Sample video files and dataset folders
├── models/                    # Trained model weights (emotion_cnn.pth, engagement_classifier.pkl)
│
├── notebooks/
│   └── engagement_analysis.ipynb   # Interactive analysis notebook
│
├── src/
│   ├── model.py               # PyTorch Custom Emotion CNN model
│   ├── train.py               # Emotion CNN training & evaluation script
│   ├── pipeline.py            # MediaPipe extraction & gaze tracking pipeline
│   ├── classifier.py          # Random Forest/SVM classifier on extracted features
│   └── create_notebook.py     # Script to generate the Jupyter Notebook
│
├── requirements.txt           # Project dependencies
└── README.md                  # Academic documentation (this file)
```

---

## 🔬 Methodology

The system processes a recorded meeting video through a multi-stage machine learning pipeline:

### 1. Face Detection & Landmark Extraction
We employ **MediaPipe Face Mesh** to detect up to 4 faces per frame and track 468+ 3D facial landmarks. The spatial arrangement of bounding boxes is used to distinguish and maintain consistent IDs for individual participants sitting in a static grid (gallery view).

### 2. Gaze Proxy Estimation
We estimate eye contact by extracting coordinates of the pupil relative to the inner and outer corners of the eyes.
* Gaze deviation is calculated as the Euclidean distance between the pupil landmark and the eye center, normalized by the eye width.
* Higher deviation values serve as a proxy for looking away from the screen.

### 3. Emotion Classification (Deep Learning)
We implement a custom **PyTorch VGG-like Convolutional Neural Network (CNN)**.
* **Input**: Grayscale crops of detected face regions resized to $48 \times 48$ pixels.
* **Architecture**: Three convolutional blocks with Batch Normalization, Max Pooling, and Dropout, followed by fully connected layers.
* **Classes**: Happy, Neutral, Confused, Bored, and Angry.

### 4. Engagement Classification (Classical Machine Learning)
Frame-level metrics are grouped into rolling windows (e.g., 30 frames/1 second). We engineer 8 feature dimensions:
* Mean face presence.
* Gaze deviation mean and variance.
* Running averages of emotion probabilities.

A **Random Forest Classifier** is trained to categorize each window as **Engaged (1)** or **Disengaged (0)**, achieving high accuracy on simulated feature distributions.

---

## 📹 Test Video & Model Assets

To test the pipeline execution immediately:

1. **MediaPipe Task Model**: Download the face landmarker file to `models/face_landmarker.task`:
   * [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

2. **Sample Test Video**: Download the Intel IoT DevKit demo video to `data/meeting.mp4`:
   * [head-pose-face-detection-female.mp4](https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female.mp4)

Run the test script to execute:
```bash
python src/test_pipeline.py
```

---

## ⚠️ Ethics, Bias, and Limitations

When presenting this project in a portfolio or academic setting, it is critical to address the inherent limitations of behavior-tracking AI:

1. **Expression $\neq$ Focus**: Inferred facial emotions do not map directly to cognitive focus or understanding. An individual with a neutral expression may be deeply engaged in thought.
2. **Demographic Bias**: Landmark models and emotion classification models can perform unevenly across different ethnicities, ages, gender identities, and physical abilities due to biases in public datasets (e.g., FER2013).
3. **Environmental Noise**: Poor room lighting, low camera resolution, and sharp head angles (tilts) drastically affect facial landmark localization.
4. **Consent & Privacy**: Real-world application of meeting analytics requires robust governance, participant consent, and data protection mechanisms.
