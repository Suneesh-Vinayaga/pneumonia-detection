import torch
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image
from build_model import build_model


def generate_gradcam(image_path, model_path, device):

    # ----------------------------
    # Load model
    # ----------------------------
    model = build_model(device, model_name="resnet")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ----------------------------
    # Select correct target layer
    # ----------------------------
    if hasattr(model, "layer4"):          # ResNet
        target_layer = model.layer4
    else:                                  # EfficientNet
        target_layer = model.features[-1]

    gradients = []
    activations = []

    # ----------------------------
    # Hook functions
    # ----------------------------
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # ----------------------------
    # Image preprocessing
    # ----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    original_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # ----------------------------
    # Forward pass
    # ----------------------------
    output = model(input_tensor)

    probs = torch.softmax(output, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)

    confidence = confidence.item()
    pred_class = pred_class.item()

    class_names = ["NORMAL", "PNEUMONIA"]
    pred_label = class_names[pred_class]

    # ----------------------------
    # Backward pass
    # ----------------------------
    model.zero_grad()
    output[0, pred_class].backward()

    # ----------------------------
    # Get gradients and activations
    # ----------------------------
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    # Global average pooling
    weights = np.mean(grads, axis=(1, 2))

    # Weighted combination
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # ----------------------------
    # Create heatmap
    # ----------------------------
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    original = cv2.resize(np.array(original_image), (224, 224))
    superimposed_img = heatmap * 0.4 + original

    # Convert properly
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # ----------------------------
    # Add prediction text
    # ----------------------------
    text = f"{pred_label} ({confidence*100:.2f}%)"

    cv2.putText(
        superimposed_img,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ----------------------------
    # Save output
    # ----------------------------
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/gradcam_result.png"
    cv2.imwrite(output_path, superimposed_img)

    print("Grad-CAM saved to outputs/gradcam_result.png")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence*100:.2f}%")

    return output_path



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = "dataset_segmented/test/PNEUMONIA/person1004_virus_1686.jpeg"
    model_path = "models/resnet_best.pth"

    generate_gradcam(image_path, model_path, device)