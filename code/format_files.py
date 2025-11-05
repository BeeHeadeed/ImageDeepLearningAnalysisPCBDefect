# Txt to Csv Parsers for the annotations and images

# Libraries
import csv
import os
from pathlib import Path
from PIL import Image




# Functions

def parse_yolo_txt(txt_file, img_width, img_height, class_names=None):
    """
    Parse a YOLO format txt file.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All coordinates are normalized (0-1)
    
    Args:
        txt_file: Path to YOLO txt file
        img_width: Original image width in pixels
        img_height: Original image height in pixels
        class_names: Optional dict mapping class_id to class name
    
    Returns:
        List of dictionaries containing parsed data
    """
    data = []
    filename = Path(txt_file).stem  # Get filename without extension
    
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert from YOLO format (normalized center coords) to corner coords
            xmin = int((x_center - width / 2) * img_width)
            ymin = int((y_center - height / 2) * img_height)
            xmax = int((x_center + width / 2) * img_width)
            ymax = int((y_center + height / 2) * img_height)
            
            # Get class name if mapping provided, otherwise use class_id
            class_name = class_names.get(class_id, str(class_id)) if class_names else str(class_id)
            
            data.append({
                'filename': filename,
                'width': img_width,
                'height': img_height,
                'class': class_name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
    
    return data


def parse_yolo_directory(directory,images_dir, class_names=None, output_csv='output.csv'):
    """
    Parse all YOLO txt files in a directory and save to CSV.
    
    Args:
        directory: Path to directory containing YOLO txt files
        images_dir: Path to directory containing corresponding images
        class_names: Optional dict mapping class_id to class name
        output_csv: Output CSV file path
    """
    all_data = []
    for folder in Path(directory).iterdir(): 
        for txt_file in Path(folder).glob('*.txt'):
            class_folder = folder.name
            image_filename = Path(images_dir) / class_folder / txt_file.with_suffix('.jpg').name

            try:
                im = Image.open(image_filename)
                w, h = im.size
            except Exception as e:
                print(f"Warning: Could not open {image_filename}: {e}")
                continue            
                
            data = parse_yolo_txt(txt_file, w, h, class_names)
            all_data.extend(data)
    
    # Write to CSV
    if all_data:
        fieldnames = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"Saved {len(all_data)} annotations to {output_csv}")
    else:
        print("No data found")



if __name__ == "__main__":

    class_names = {
        0: 'Missing_hole',
        1: 'Mouse_bite',
        2: 'Open_circuit',
        3: 'Short',
        4: 'Spur',
        5: 'Supurious_copper'
    }
    
    parse_yolo_directory(
        directory='Labeled_images',
        class_names=class_names,
        images_dir= '.\\data\\images',
        output_csv='data\\annotations.csv'
    )


