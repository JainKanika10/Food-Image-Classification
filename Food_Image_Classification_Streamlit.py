import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from torchvision.datasets import ImageFolder

# -----------------------------
# 1️⃣ Setup: Device, Model, Classes
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet model
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 101)  # adjust to your num_classes
model.load_state_dict(torch.load("best_model1.pth", map_location=device))
model = model.to(device)
model.eval()

# Load class names
dataset_path = "C:\\Users\\jainka\\OneDrive - Hewlett Packard Enterprise\\DS-C-WE-E-B57\\Food Image Classification\\food-101\\train"
dataset = ImageFolder(root=dataset_path)
class_idx_to_name = {idx: name for idx, name in enumerate(dataset.classes)}

# -----------------------------
# 2️⃣ Grad-CAM hooks
# -----------------------------
target_layer = model.layer4[-1].conv3
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# -----------------------------
# 3️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="ResNet Food Classifier", layout="wide")
st.title("🍔 ResNet Food Classification with Grad-CAM")
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of food.  
2. The model predicts the class.  
3. Grad-CAM shows which regions influenced the prediction.
""")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_name = uploaded_file.name
    st.sidebar.write(f"**Image Name:** {img_name}")
    
    img = Image.open(uploaded_file).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Forward pass
    output = model(input_tensor)
    pred_class_idx = output.argmax().item()
    pred_class_name = class_idx_to_name[pred_class_idx]
    
    st.sidebar.write(f"**Predicted Class:** {pred_class_name} ({pred_class_idx})")
    
    # Backward pass for Grad-CAM
    model.zero_grad()
    output[0, pred_class_idx].backward()
    
    gradients_mean = gradients.mean(dim=[2,3], keepdim=True)
    cam = (gradients_mean * activations).sum(dim=1).squeeze()
    cam = np.maximum(cam.detach().cpu().numpy(), 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = np.uint8(heatmap * 0.4 + np.array(img) * 0.6)
    
    # Display images side by side
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(cam, caption="Grad-CAM Heatmap", use_container_width=True)
    with col3:
        st.image(superimposed, caption="Overlay (Grad-CAM)", use_container_width=True)
