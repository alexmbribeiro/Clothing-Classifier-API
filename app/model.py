import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from huggingface_hub import hf_hub_download


NUM_CLASSES = 10
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def load_model():
    
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model_path = hf_hub_download(repo_id="alexrmb/fashion-classifier-model", filename="fashion_model.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),  # ensure 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    return transform(image).unsqueeze(0)  # add batch dimension

# Prediction function
def predict(model, image: Image.Image):
    tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]
