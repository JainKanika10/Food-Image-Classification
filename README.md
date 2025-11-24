# 🍽️ Food-101 Image Classification using Transfer Learning

This project focuses on classifying food images into **101 categories** using transfer learning techniques with **ResNet-50** and **EfficientNet-B0**.
The implementation includes dataset preparation, model training, evaluation, and model interpretation using Grad-CAM visualizations.

---

## 📌 Problem Statement

Develop a deep learning model to classify images into **101 food categories** using pretrained model architectures.

The project includes:

- ✔ Data loading and preprocessing  
- ✔ Training with validation monitoring  
- ✔ Evaluation using macro F1-score  
- ✔ Optional explainability using Grad-CAM  

---

## 📂 Dataset: Food-101

📎 Dataset Link: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

### Dataset Summary

| Property | Value |
|---------|-------|
| Total Images | ~101,000 |
| Classes | 101 |
| Training Images/Class | 750 |
| Test Images/Class | 250 |

### Folder Structure

```
food-101/
 ├── train/
 │     ├── apple_pie/
 │     ├── baby_back_ribs/
 │     └── ...
 └── test/
       ├── apple_pie/
       ├── baby_back_ribs/
       └── ...
```

---

## 🎯 Real-World Use Cases

- Restaurant menu automation  
- Food delivery platform image tagging  
- Diet monitoring and calorie estimation  
- Visual search and recommendation systems  

---

## 🛠️ Approach

### 1️⃣ Data Preparation

- Download and extract dataset  
- Organize into `train/` and `test/` structure  
- Normalize using ImageNet mean and std  

### 2️⃣ Exploratory Data Analysis (EDA)

- Visualize samples  
- Inspect class imbalance  
- Analyze image resolution variance  

### 3️⃣ Data Augmentation

Techniques used:

- Random rotation  
- Horizontal flip  
- Color jitter  
- RandomResizedCrop  

### 4️⃣ Model Selection: Transfer Learning

| Model | Method |
|-------|--------|
| ResNet-50 | Replace fully connected layer |
| EfficientNet-B0 | Replace classifier head |

### 5️⃣ Training Pipeline

- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** AdamW  
- **Scheduler:** ReduceLROnPlateau / StepLR  
- Saved best model checkpoint based on validation score  

---

## 📊 Evaluation Metrics

Measured using:

- Macro F1-score  
- Classification report  
- Per-class accuracy  
- Confusion matrix visualization  

---

## 🔍 Explainability (Optional)

Generated **Grad-CAM heatmaps** to visualize feature-based model attention on images.

---

## 🚀 Deployment (Optional)

- Export model through **TorchScript**
- Create user interface with **Streamlit**

Features:

- Image upload  
- Real-time predictions  
- Grad-CAM overlay display  

---

## ⚙ Installation & Setup

### Create Virtual Environment

```bash
# Windows
python -m venv food_env
food_env\Scripts\activate


### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib seaborn scikit-learn pandas streamlit pillow opencv-python
```

---

## 📁 Project Structure

```
Food101-Project/
├── train_food101.py
├── best_model1.pth
├── Food_Image_Classification_Streamlit.py
├── class_idx_to_name.json
├── README.md
└── results/
```

---

## ▶️ Usage

### Model Training

### Streamlit App

```bash
streamlit run Food_Image_Classification_Streamlit.py
```

---

## 📈 Evaluation & Results

- Macro F1-score reported on test set  
- Confusion matrix visualized  
- Grad-CAM outputs used for interpretation  

---

## 🔁 Reproducibility Checklist

- Download Food-101 dataset  
- Create folder structure  
- Train model or load checkpoint  
- Run inference or UI app  

---

## 🏷 Technical Tags

`Machine Learning`, `Deep Learning`, `Computer Vision`, `PyTorch`,  
`CNN`, `ResNet50`, `EfficientNet`, `Transfer Learning`, `Food-101`,  
`Grad-CAM`, `Streamlit`, `TorchScript`.

---

### ✅ End of README
