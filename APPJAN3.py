import streamlit as st
from streamlit_autorefresh import st_autorefresh
from PIL import Image
import numpy as np
import torch
import torchvision
import time

from torchvision.models import resnet50, ResNet50_Weights

# Load the pre-trained ResNet50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Function to preprocess the image
def preprocess_image(img):
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(img).unsqueeze(0)

# Function to make predictions and get top 5 probabilities
def predict(model, img):
    with torch.no_grad():
        output = model(img)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        top_p, top_class = probs.topk(5)
        return top_p.numpy(), top_class.numpy()

# Create the Streamlit app
st.title("Image Classifier")
count = st_autorefresh(interval=4000, limit=100, key="fizzbuzzcounter")

directory1 = "Images"
import os

directory = os.fsencode(directory1)


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # print(os.path.join(directory, filename))
        image = Image.open("Images/"+filename)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make a prediction
        probs, classes = predict(model, processed_image)

        # Get the class names from the ImageNet labels
        labels_path = 'ImageNetLabels.txt'
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Display the top 5 predictions
        for p, c in zip(probs, classes):
            st.write(f"{labels[c]:20}: {p:.2f}")
        continue

