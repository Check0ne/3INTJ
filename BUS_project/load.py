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

import numpy as np
import matplotlib.pyplot as plt
import PIL

plt.subplots(3, 3, figsize=(8, 8))
for i, k in enumerate(np.random.randint(num_total, size=9)):
    im = PIL.Image.open(image_files_list[k])
    arr = np.array(im)
    plt.subplot(3, 3, i + 1)
    plt.xlabel(class_names[image_class[k]])
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
plt.tight_layout()
plt.show()
print("Hello") 

# define transforms for image and segmentation
images = image_files_list
segs = seg_files_list

import torch
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

from torch.utils.tensorboard import SummaryWriter

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
val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()]) #  validation에서는 crop, rotate를 하지 않는다.

# define array dataset, data loader
check_ds = ArrayDataset(images, train_imtrans, segs, train_segtrans)
check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im, seg = monai.utils.misc.first(check_loader)
print(im.shape, seg.shape)
print("Hello")

# shuffle
val_frac = 0.15
test_frac = 0.15
length = len(images)
indices = np.arange(length)
np.random.shuffle(indices)

images = [images[i] for i in indices] ### shuffled list
segs = [segs[i] for i in indices] ### shuffled list

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]


#train_x = [image_files_list[i] for i in train_indices]
#train_y = [image_class[i] for i in train_indices]
#val_x = [image_files_list[i] for i in val_indices]
#val_y = [image_class[i] for i in val_indices]
#test_x = [image_files_list[i] for i in test_indices]
#test_y = [image_class[i] for i in test_indices]

print(f"Training count: {len(train_indices)}, Validation count: " f"{len(val_indices)}, Test count: {len(test_indices)}")

# create a training data loader --> 그 전에 섞어야 할 것 같은데...
train_ds = ArrayDataset(images[:20], train_imtrans, segs[:20], train_segtrans)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
# create a validation data loader
val_ds = ArrayDataset(images[-20:], val_imtrans, segs[-20:], val_segtrans)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])