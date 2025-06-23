import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def convert_coco_to_yolo(coco_json_path, output_dir, images_dir):
    """
    Converts COCO segmentation annotations to YOLO bounding box format.
    Creates .txt files for each image.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping for image ID to image info (width, height)
    image_info = {img['id']: {'width': img['width'], 'height': img['height'], 'file_name': img['file_name']}
                  for img in coco_data['images']}

    # Create a mapping for category ID to category name
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    class_names = [cat['name'] for cat in coco_data['categories']]

    # We need the class names in a sorted order to create the `names` list in data.yaml
    # Ultralytics often uses alphabetical order for class IDs unless explicitly mapped.
    # It's safest to create a `class_to_id` mapping based on sorted names later.
    sorted_class_names = sorted(list(set(class_names)))
    class_to_id = {name: i for i, name in enumerate(sorted_class_names)}

    print(f"Converting annotations from {coco_json_path}...")
    for annotation in tqdm(coco_data['annotations']):
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        segmentation = annotation['segmentation'] # This is a list of polygons
        bbox = annotation['bbox'] # [x, y, width, height] in absolute pixels

        img_width = image_info[image_id]['width']
        img_height = image_info[image_id]['height']
        img_filename = image_info[image_id]['file_name']

        # Get the class name and map it to the new integer ID
        class_name = category_map[category_id]
        yolo_class_id = class_to_id[class_name]

        # Convert COCO bbox [x_min, y_min, width, height] to YOLO [x_center, y_center, width, height] (normalized)
        x_center = (bbox[0] + bbox[2] / 2) / img_width
        y_center = (bbox[1] + bbox[3] / 2) / img_height
        norm_width = bbox[2] / img_width
        norm_height = bbox[3] / img_height

        # Create the YOLO .txt file path
        # Assuming images are in a structure like images/train/image.jpg
        # and labels should go into labels/train/image.txt
        img_name_without_ext = Path(img_filename).stem
        output_txt_path = Path(output_dir) / f"{img_name_without_ext}.txt"

        with open(output_txt_path, 'a') as f: # 'a' to append if multiple objects in one image
            f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

    return sorted_class_names

if __name__ == "__main__":
    dataset_root = "food_recognition_2022_dataset" # Adjust this path

    # Process training set
    train_coco_json = Path(dataset_root) / "train" / "annotations" / "instances_train.json"
    train_images_dir = Path(dataset_root) / "train" / "images"
    train_labels_output_dir = Path(dataset_root) / "labels" / "train"
    train_class_names = convert_coco_to_yolo(train_coco_json, train_labels_output_dir, train_images_dir)

    # Process validation set
    val_coco_json = Path(dataset_root) / "val" / "annotations" / "instances_val.json"
    val_images_dir = Path(dataset_root) / "val" / "images"
    val_labels_output_dir = Path(dataset_root) / "labels" / "val"
    val_class_names = convert_coco_to_yolo(val_coco_json, val_labels_output_dir, val_images_dir)

    # Process test set (optional, you can convert it for testing after training)
    test_coco_json = Path(dataset_root) / "test" / "annotations" / "instances_test.json"
    test_images_dir = Path(dataset_root) / "test" / "images"
    test_labels_output_dir = Path(dataset_root) / "labels" / "test"
    test_class_names = convert_coco_to_yolo(test_coco_json, test_labels_output_dir, test_images_dir)

    # Verify that class names are consistent across splits (they should be)
    if train_class_names != val_class_names or train_class_names != test_class_names:
        print("Warning: Class names are inconsistent across train/val/test splits.")
    
    # Save class names to a file for easy reference
    with open("food_classes.txt", "w") as f:
        for name in train_class_names:
            f.write(f"{name}\n")

    print(f"Total classes: {len(train_class_names)}")
    print("Annotation conversion complete!")