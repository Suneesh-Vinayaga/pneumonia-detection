import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

from build_model import build_model
from lung_segmentation import segment_lung
from gradcam import generate_gradcam


# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Load Model
# -----------------------
model_path = "models/efficientnet_best.pth"

model = build_model(device, model_name="efficientnet")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# -----------------------
# Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# -----------------------
# Prediction Function
# -----------------------
def predict(image_path):

    # Lung segmentation
    _, _, segmented = segment_lung(image_path)

    # Convert to PIL
    img = Image.fromarray(segmented).convert("RGB")

    # Transform
    img = transform(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    classes = ["NORMAL", "PNEUMONIA"]

    prediction = classes[pred.item()]
    confidence = confidence.item()

    # Generate Grad-CAM heatmap
    heatmap_path = generate_gradcam(image_path, model_path, device)

    return prediction, confidence, heatmap_path


# -----------------------
# Raw Prediction (No Segmentation)
# -----------------------
def predict_raw(image_path):

    # Load original image directly (no segmentation)
    img = Image.open(image_path).convert("RGB")

    # Transform
    img = transform(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    classes = ["NORMAL", "PNEUMONIA"]

    prediction = classes[pred.item()]
    confidence = confidence.item()

    return prediction, confidence


# -----------------------
# Run Test
# -----------------------
if __name__ == "__main__":

    image_path = "dataset/test/PNEUMONIA/person1_virus_12.jpeg"

    prediction, confidence, heatmap_path = predict(image_path)

    print("Prediction:", prediction)
    print("Confidence:", round(confidence*100,2), "%")
    print("Heatmap:", heatmap_path)