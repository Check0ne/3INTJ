import os
from PIL import Image
from glob import glob

data_dir = '/workspace/Dataset_BUSI/Dataset_BUSI_with_GT'

class_names = ['benign', 'malignant', 'normal']
num_class = len(class_names)

image_files = [
    sorted(glob(os.path.join(data_dir, class_names[i], "*).png")))
    for i in range (num_class)
]

seg_files = [
    sorted(glob(os.path.join(data_dir, class_names[i], "*mask.png")))
    for i in range (num_class)
]

num_each = [len(image_files[i]) for i in range(num_class)] # class 별 image_files 개수 리스트

image_files_list = [] 
seg_files_list = []
image_class = []

for i in range(num_class): # class 개수 만큼
    image_files_list.extend(image_files[i]) # image_files[i] 리스트의 요소(image)를 image_files_list에 추가
    seg_files_list.extend(seg_files[i])
    image_class.extend([i] * num_each[i]) # class 이름 리스트 생성 [A, A, A, ... B, B, B, ... C, C, C ...] class 요소의 개수는 num_each 만큼 
num_total = len(image_class) # image_class의 요소 개수
image_width, image_height = Image.open(image_files_list[0]).size # 하나의 이미지 열어서 width, height 값 추출

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")

########################################################
# define transforms for image and segmentation
images = image_files_list
segs = seg_files_list

import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
from monai.visualize import plot_2d_or_3d_image

train_imtrans = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        RandSpatialCrop((96, 96), random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    ]
)
train_segtrans = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        RandSpatialCrop((96, 96), random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    ]
)
val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
# define array dataset, data loader
check_ds = ArrayDataset(images, train_imtrans, segs, train_segtrans)