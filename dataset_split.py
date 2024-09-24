import json
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# load images from a folder
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            filenames.append(file)
    return np.array(images), filenames

# save images to a folder
def save_images_to_folder(imgs, filenames, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for img, filename in zip(imgs, filenames):
        img_path = os.path.join(folder_name, filename)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_bgr)

# split the COCO annotations
def split_coco_annotations(annotations_file, filenames_train, filenames_val, output_dir):
    with open(annotations_file, 'r') as f:
        coco = json.load(f)
    
    train_annotations = {key: [] if isinstance(coco[key], list) else coco[key] for key in coco}
    val_annotations = {key: [] if isinstance(coco[key], list) else coco[key] for key in coco}
    
    image_id_map = {}
    for img in coco['images']:
        if img['file_name'] in filenames_train:
            train_annotations['images'].append(img)
            image_id_map[img['id']] = 'train'
        elif img['file_name'] in filenames_val:
            val_annotations['images'].append(img)
            image_id_map[img['id']] = 'val'
    
    for ann in coco['annotations']:
        if image_id_map[ann['image_id']] == 'train':
            train_annotations['annotations'].append(ann)
        else:
            val_annotations['annotations'].append(ann)
    
    with open(os.path.join(output_dir, 'train_annotations.json'), 'w') as f:
        json.dump(train_annotations, f, indent=4)
    
    with open(os.path.join(output_dir, 'val_annotations.json'), 'w') as f:
        json.dump(val_annotations, f, indent=4)



input_folder = "Pedestrian_dataset_for_internship_assignment"
annotations_file = "random_sample_mavi_2_gt.json"
output_dir = "dataset"

images, filenames = load_images_from_folder(input_folder)


X_train, X_val, filenames_train, filenames_val = train_test_split(images, filenames, test_size=0.2, random_state=42)


save_images_to_folder(X_train, filenames_train, os.path.join(output_dir, "train"))
save_images_to_folder(X_val, filenames_val, os.path.join(output_dir, "val"))

split_coco_annotations(annotations_file, filenames_train, filenames_val, output_dir)
