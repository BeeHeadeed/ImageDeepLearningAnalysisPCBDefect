from image_preproccesing_functions import resolve_enhanced_labels,resolve_enhanced_yaml

if __name__ == "__main__":
    resolve_enhanced_yaml()
    resolve_enhanced_labels()
    import os, shutil

    ENHANCED_ROOT = "data/enhanced_images"

    # 1 Move train/val/test folders into an 'images/' subfolder
    images_folder = os.path.join(ENHANCED_ROOT, "images")
    os.makedirs(images_folder, exist_ok=True)

    for split in ["train", "val", "test"]:
        src = os.path.join(ENHANCED_ROOT, split)
        dst = os.path.join(images_folder, split)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {src} → {dst}")

    print("Enhanced images restructured under 'images/' folder.")
