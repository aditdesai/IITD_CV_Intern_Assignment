import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os

coco = COCO("dataset/annotations/instances_train2017.json")
images_path = "dataset/train2017"

# visualize ground truth bounding box
def vis_bbox(id):
    img_info = coco.imgs[id]
    img_path = os.path.join(images_path, img_info['file_name'])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ann_ids = coco.getAnnIds(imgIds=id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        x, y, w, h = map(int, ann['bbox'])
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)

    plt.imshow(img)
    plt.show()

vis_bbox(66)