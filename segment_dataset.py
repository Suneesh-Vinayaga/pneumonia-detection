import os
import cv2
from tqdm import tqdm

# import the segmentation function you already wrote
from lung_segmentation import segment_lung


# original dataset
INPUT_DATASET = "dataset"

# output dataset
OUTPUT_DATASET = "dataset_segmented"


def process_folder(split):

    input_path = os.path.join(INPUT_DATASET, split)
    output_path = os.path.join(OUTPUT_DATASET, split)

    os.makedirs(output_path, exist_ok=True)

    for class_name in os.listdir(input_path):

        class_input = os.path.join(input_path, class_name)
        class_output = os.path.join(output_path, class_name)

        os.makedirs(class_output, exist_ok=True)

        images = os.listdir(class_input)

        print(f"\nProcessing {split}/{class_name} ({len(images)} images)")

        for img_name in tqdm(images):

            img_path = os.path.join(class_input, img_name)

            try:
                original, mask, segmented = segment_lung(img_path)

                save_path = os.path.join(class_output, img_name)

                cv2.imwrite(save_path, segmented)

            except Exception as e:
                print(f"Skipping {img_name}: {e}")


if __name__ == "__main__":

    for split in ["train", "val", "test"]:
        process_folder(split)

    print("\nSegmentation complete.")