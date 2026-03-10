import cv2
import torch
import numpy as np
import torchxrayvision as xrv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load pretrained lung segmentation model
model = xrv.baseline_models.chestx_det.PSPNet()
model = model.to(device)
model.eval()


def segment_lung(image_path):

    # read image in grayscale
    img = cv2.imread(image_path, 0)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    original = img.copy()

    # normalize using torchxrayvision normalization
    img = xrv.datasets.normalize(img, 255)

    # resize for model input
    img = cv2.resize(img, (512, 512))

    # convert to float32
    img = img.astype(np.float32)

    # convert to tensor
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        pred = model(x)

    # convert prediction to numpy
    pred = pred.cpu().numpy()

    # extract left lung (channel 4) and right lung (channel 5) masks
    left_lung = pred[0, 4]
    right_lung = pred[0, 5]

    # combine both lung masks
    mask = np.maximum(left_lung, right_lung)

    # threshold mask
    mask = (mask > 0.5).astype(np.uint8)

    # resize mask back to original image size
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    # apply mask
    segmented = original * mask

    return original, mask, segmented


if __name__ == "__main__":

    image_path = r"dataset/test/PNEUMONIA/person1_virus_12.jpeg"

    original, mask, segmented = segment_lung(image_path)

    cv2.imshow("Original", original)
    cv2.imshow("Mask", mask * 255)
    cv2.imshow("Segmented Lung", segmented)

    cv2.waitKey(0)
    cv2.destroyAllWindows()