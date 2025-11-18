# Brain-Tumor-Classification-CNN
Brain Tumor Classification using Convolutional Neural Networks (CNN) A deep learning-based system that classifies brain MRI scans into glioma, meningioma, pituitary tumor, and no tumor using a custom CNN model. Includes Grad-CAM explainability and a Streamlit web app for real-time image prediction.
🧠 Brain Tumor MRI Classification using Convolutional Neural Networks (CNN)

This project builds a complete AI system that classifies Brain MRI scans into four categories using a custom Convolutional Neural Network (CNN) architecture. It includes model training, evaluation, explainability using Grad-CAM, and a fully interactive Streamlit web application for real-time predictions.

🚀 Project Overview

Brain tumors are one of the most critical neurological conditions, and MRI-based detection plays a major role in early diagnosis.
This project automates tumor classification using deep learning techniques, making diagnosis faster and more accessible.

The system detects the following classes:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

The project demonstrates every major stage of an ML pipeline:

✔ Data Loading & Preprocessing
✔ Data Augmentation
✔ Custom CNN Model Development
✔ Training & Validation
✔ Model Evaluation
✔ Grad-CAM Explainability
✔ Web App Deployment (Streamlit)

📂 Project Structure
Brain-Tumor-Classification/
│── app.py                        # Streamlit web app
│── BrainTumor.ipynb              # Jupyter Notebook with full code
│── brain_tumor_cnn_model.h5      # Saved CNN model
│── class_labels.npy              # Saved class names
│── requirements.txt              # Dependencies for running the app
│── images/
│     ├── accuracy_curve.png
│     ├── loss_curve.png
│     ├── confusion_matrix.png
│     ├── gradcam_example.png
│
└── README.md

🗂 Dataset Description

The dataset contains MRI images categorized into four tumor types.

Folder structure:

Training/
    glioma_tumor/
    meningioma_tumor/
    pituitary_tumor/
    no_tumor/

Testing/
    glioma_tumor/
    meningioma_tumor/
    pituitary_tumor/
    no_tumor/


Images are preprocessed by:

Resizing to 224 × 224

Normalizing pixel values (0–255 → 0–1)

Label encoding + one-hot encoding

Shuffling

🛠 Model Architecture

The CNN model includes:

Data Augmentation (Flip, Rotation, Zoom, Translation)

4 Convolution Blocks

Batch Normalization

MaxPooling

Global Average Pooling

Dense layers with Dropout

Softmax output layer (4 classes)

Frameworks Used:

TensorFlow / Keras

NumPy

OpenCV

Matplotlib & Seaborn

Streamlit

📊 Model Performance
Metric	Score
Test Accuracy: 0.40
Test Precision: 0.36
Test Recall: 0.44
Test F1 Score: 0.33
📈 Training Curves
accuracy_curve.png
loss_curve.png
🔥 Confusion Matrix

Shows class-wise prediction performance.

images/confusion_matrix.png

🔍 Grad-CAM Explainability

Grad-CAM visualizes which regions of the MRI contributed to the model’s prediction.
Useful for building trust in medical AI.

🌐 Streamlit Web App

This project includes a fully functional Streamlit application.

▶ Run the app:
streamlit run app.py

🧩 Features:

MRI Image Upload

Tumor Type Prediction

Confidence Score

Grad-CAM Heatmap for Explainability

Clean and User-Friendly Interface

🔧 Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/yourusername/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Web App
streamlit run app.py

🧪 Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Streamlit

🎯 Conclusion

This project provides a complete deep-learning pipeline for brain tumor detection using MRI images.
It includes training, evaluation, explainability, and deployment — making it an ideal project for medical AI research, ML learning, and portfolio building.

💡 Possible Improvements

Use Transfer Learning (VGG16, ResNet50, MobileNetV2)

Add segmentation-based localization

Convert model into TensorFlow Lite for mobile apps

🤝 Contributions

Pull requests and improvements are welcome!

📬 Contact

For queries or collaboration, feel free to reach out!
