import os
import cv2
import yaml
import glob
import numpy as np
from PIL import Image
import shutil
from pathlib import Path

# ===============================
# === Image Processing Helpers ==
# ===============================

def Image_rescaling(img_object):
    if isinstance(img_object, Image.Image):
        img_array = np.array(img_object)
    else:
        img_array = img_object

    if np.issubdtype(img_array.dtype, np.integer) and img_array.max() > 1:
        return img_array / 255.0
    else:
        return img_array

def Image_contrased_enhancemend(img_object):
    img_Lab = cv2.cvtColor(img_object, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_Lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    img_Lab_clahe = cv2.merge([l_clahe, a, b])
    img_enhanced = cv2.cvtColor(img_Lab_clahe, cv2.COLOR_LAB2BGR)
    return img_enhanced

def Image_convertion_Lab(img_object):
    return cv2.cvtColor(img_object, cv2.COLOR_BGR2LAB)

def Image_convertion_HSV(img_object):
    return cv2.cvtColor(img_object, cv2.COLOR_BGR2HSV)

def remove_noise(img, method="bilateral"):
    if img is None:
        raise ValueError("Image not found or invalid path")

    if method == "gaussian":
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
    elif method == "median":
        denoised = cv2.medianBlur(img, 5)
    elif method == "bilateral":
        denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == "fastNlMeans":
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    else:
        raise ValueError("Invalid method. Choose from: 'gaussian', 'median', 'bilateral', 'fastNlMeans'.")
    return denoised

def Image_Processing_main(file_path: str):
    try:
        img_original = cv2.imread(file_path)
        if img_original is None:
            raise FileNotFoundError(f"Image not loaded: {file_path}")

        img_enhanced = Image_contrased_enhancemend(img_original)
        img_denoised = remove_noise(img_enhanced)
        img_lab = Image_convertion_Lab(img_denoised)
        img_hsv = Image_convertion_HSV(img_denoised)
        img_rescaled_RGB = Image_rescaling(img_denoised)

        return img_original, img_rescaled_RGB, img_enhanced, img_lab, img_hsv

    except FileNotFoundError:
        print(f"Error: File not found or could not be loaded at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred processing {file_path}: {e}")
        return None

# =================================
# === Dataset Preprocessing Tool ==
# =================================

def preprocess_dataset_images(yaml_file):
    """Reads dataset.yaml and enhances all images in train/val/test."""
    with open(yaml_file, 'r') as f:
        ds = yaml.safe_load(f)

    for split in ("train", "val", "test"):
        if split not in ds:
            continue

        split_path = ds[split]
        if not os.path.isabs(split_path):
            base_path = os.path.dirname(yaml_file)
            split_path = os.path.join(base_path, split_path)

        if not os.path.exists(split_path):
            print(f"  Skipping {split}: path not found → {split_path}")
            continue

        print(f"\n🔹 Processing split: {split} ({split_path})")
        files = glob.glob(os.path.join(split_path, "**", "*.*"), recursive=True)
        out_root = os.path.join("data", "enhanced_images", split)
        os.makedirs(out_root, exist_ok=True)

        count = 0
        for img_path in files:
            if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            result = Image_Processing_main(img_path)
            if result is None:
                continue
            _, _, img_enhanced, _, _ = result
            out_path = os.path.join(out_root, os.path.basename(img_path))
            Image.fromarray(cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)).save(out_path)
            count += 1

        print(f" {count} images enhanced and saved to {out_root}")

def resolve_enhanced_yaml():
  original_yaml =os.path.join("data","yolo_dataset","dataset.yaml")
  enhanced_yaml = os.path.join("data","enhanced_images","dataset_enhanced.yaml")

  # Load the original YAML
  with open(original_yaml, 'r') as f:
      data = yaml.safe_load(f)

  # Replace paths for train/val/test
  for split in ["train", "val", "test"]:
      data[split] = os.path.join("data", "enhanced_images", split)

  # Save to a new YAML file
  with open(enhanced_yaml, 'w') as f:
      yaml.dump(data, f, default_flow_style=False)

print(f" Enhanced YAML created!")

def resolve_enhanced_labels(): 
  # Original YOLO labels
  labels_root = Path("data/yolo_dataset/labels")

  # Enhanced images
  enhanced_root = Path("data/enhanced_images")

  # Loop over splits
  for split in ["train", "val", "test"]:
      img_dir = enhanced_root / split
      for img_path in img_dir.glob("*.*"):
          # Corresponding label filename
          label_file = labels_root / split / f"{img_path.stem}.txt"
          if label_file.exists():
              shutil.copy2(label_file, img_path.parent / label_file.name)
          else:
              print(f" Label missing for {img_path.name}")

  print(" Labels copied next to enhanced images")


# ===========================================
# === Run Preprocessing on Your Dataset =====
# ===========================================

if __name__ == "__main__":
    YAML_PATH = os.path.join("data","yolo_dataset","dataset.yaml" ) 
    preprocess_dataset_images(YAML_PATH)
    resolve_enhanced_yaml()
    resolve_enhanced_labels()
  