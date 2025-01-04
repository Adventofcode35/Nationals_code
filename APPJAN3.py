import streamlit as st
from streamlit_autorefresh import st_autorefresh
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import os
import time

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
st.title("Image Classifier For Railways")

placeholder = st.empty()
placeholder2 = st.empty()

directory1 = "Images"
directory = os.fsencode(directory1)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Process for each image
        image = Image.open("Images/" + filename)
        placeholder.image(image, caption='Uploaded Image', use_container_width=True)
        processed_image = preprocess_image(image)
        probs, classes = predict(model, processed_image)

        # Get the class names from the ImageNet labels
        labels_path = 'ImageNetLabels.txt'
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]


        # Display the top 5 predictions in placeholder2
        string = ""
        for p, c in zip(probs, classes):
            string += f"{labels[c]:20}: {p:.2f}\n"
        placeholder2.text(string)

        # Add a small delay for better visualization
        time.sleep(4) 

        # Clear both placeholders for the next image
        placeholder.empty()
        placeholder2.empty()  # Uncomment this if you want to clear after each image
        