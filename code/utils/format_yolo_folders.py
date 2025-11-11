import pandas as pd
import shutil
from pathlib import Path
from collections import defaultdict

def create_yolo_structure(
    images_dir='data/images',
    train_csv='data/train_annotations.csv',
    val_csv='data/val_annotations.csv',
    test_csv='data/test_annotations.csv',
    output_dir='data/yolo_dataset',
    class_mapping=None
):
    """
    Reorganize data from CSV format to YOLO folder structure.
    
    Structure created:
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
    
    Args:
        images_dir: Path to Images directory with category subfolders
        train_csv, val_csv, test_csv: Paths to split CSV files
        output_dir: Output directory for YOLO structure
        class_mapping: Dict mapping class names to class IDs (optional)
    """
    
    output_path = Path(output_dir)
    images_path = Path(images_dir)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = {
        'train': train_csv,
        'val': val_csv,
        'test': test_csv
    }
    
    # Auto-generate class mapping if not provided

    if class_mapping is None:
        all_classes = set()
        for csv_file in splits.values():
            df = pd.read_csv(csv_file)
            all_classes.update(df['class'].unique())
        class_mapping = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
        print(f"Auto-generated class mapping: {class_mapping}")
    
    stats = defaultdict(lambda: {'images': 0, 'annotations': 0})
    
    for split_name, csv_file in splits.items():
        print(f"\nProcessing {split_name} split...")
        df = pd.read_csv(csv_file)
        
        # Group annotations by filename
        grouped = df.groupby('filename')
        
        for filename, group in grouped:
            # Find source image in category folders
            source_image = None
            for category_folder in images_path.iterdir():
                if not category_folder.is_dir():
                    continue
                
                # Try different extensions
                for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    potential_path = category_folder / f"{filename}{ext}"
                    if potential_path.exists():
                        source_image = potential_path
                        break
                if source_image:
                    break
            
            if source_image is None:
                print(f"  Warning: Could not find image for {filename}")
                continue
            
            # Copy image to YOLO structure
            dest_image = output_path / 'images' / split_name / source_image.name
            shutil.copy2(source_image, dest_image)
            
            # Create YOLO format label file
            label_file = output_path / 'labels' / split_name / f"{filename}.txt"
            
            with open(label_file, 'w') as f:
                for _, row in group.iterrows():
                    # Get class ID
                    class_id = class_mapping[row['class']]
                    
                    # Convert bounding box to YOLO format (normalized center coords)
                    width = row['width']
                    height = row['height']
                    
                    x_center = ((row['xmin'] + row['xmax']) / 2) / width
                    y_center = ((row['ymin'] + row['ymax']) / 2) / height
                    bbox_width = (row['xmax'] - row['xmin']) / width
                    bbox_height = (row['ymax'] - row['ymin']) / height
                    
                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                    
                    stats[split_name]['annotations'] += 1
            
            stats[split_name]['images'] += 1
        
        print(f"   Processed {stats[split_name]['images']} images with {stats[split_name]['annotations']} annotations")
    
    # Create dataset.yaml file
    sorted_classes = [name for name, _id in sorted(class_map.items(), key=lambda x: x[1])]

    yaml_content = f"""# YOLO Dataset Configuration
                        path: {output_path.absolute()}
                        train: images/train
                        val: images/val
                        test: images/test

                        # Classes
                        nc: {len(sorted_classes)}
                        names: {sorted_classes}
                        """
    
    yaml_file = output_path / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Created dataset.yaml at {yaml_file}")
    print("\n=== Summary ===")
    for split_name, split_stats in stats.items():
        print(f"{split_name.upper()}: {split_stats['images']} images, {split_stats['annotations']} annotations")
    
    print(f"\n=== Class Mapping ===")
    for class_name, class_id in class_mapping.items():
        print(f"  {class_id}: {class_name}")
    
    return output_path

if __name__ == "__main__":

    class_map = {
        'Missing_hole': 0,
        'Mouse_bite': 2,
        'Open_circuit': 3,
        'Short': 1
        }

    
    output_dir = create_yolo_structure(class_mapping=class_map)
