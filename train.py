'''
수정해야 할 사항
1. mask 병합
2. input image로 cropped image 대신 resized image 사용
3. validation, test dataset transform 확인
4. UNet input, output channel 확인....
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from glob import glob

# data load
data_dir = '/workspace/Dataset_BUSI/Dataset_BUSI_with_GT'

class_names = ['benign', 'malignant', 'normal']
num_class = len(class_names)

image_files = [
    sorted(glob(os.path.join(data_dir, class_names[i], "*).png")))
    for i in range (num_class)
]

seg_files = [
    sorted(glob(os.path.join(data_dir, class_names[i], "*mask.png"))) # mask가 여러 개인 image의 경우, 첫 번째 mask만 불러오고 있다. --> mask 병합이 필요하다.
    for i in range (num_class)
]

print(len(image_files[0]), len(image_files[1]), len(image_files[2]))
print(len(seg_files[0]), len(seg_files[1]), len(seg_files[2]))

num_each = [len(image_files[i]) for i in range(num_class)] # class 별 image_files 개수 리스트

image_files_list = [] 
seg_files_list = []
image_class = []

for i in range(num_class): 
    image_files_list.extend(image_files[i]) # image_files[i] 리스트의 요소(image)를 image_files_list에 추가
    seg_files_list.extend(seg_files[i])
    image_class.extend([i] * num_each[i]) # class 이름 리스트 생성 [A, A, A, ... B, B, B, ... C, C, C ...] class 요소의 개수는 num_each 
num_total = len(image_class) # image_class의 요소 개수
image_width, image_height = Image.open(image_files_list[0]).size # 하나의 이미지 열어서 width, height 값 추출

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")

plt.subplots(3, 3, figsize=(8, 8))
for i, k in enumerate(np.random.randint(num_total, size=9)):
    im = PIL.Image.open(seg_files_list[500])
    arr = np.array(im)
    plt.subplot(3, 3, i + 1)
    plt.xlabel(class_names[image_class[k]])
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
plt.tight_layout()
plt.show()

# normal_mask만 numpy arry가 3차원이다. 기존의 channel=3을 없애고 다시 channel=1을 추가했다. transform까지 정상적으로 되는 것을 확인할 수 있다.
# 기존에는 MONAI에서 파일명으로 접근했지만, 배열을 수정해야 하기 때문에 파일을 열어서 배열을 수정하고, 수정한 배열을 리스트에 넣는다.
images = [np.array(PIL.Image.open(i)) for i in image_files_list]
segsorigin = [np.array(PIL.Image.open(i)) for i in seg_files_list]
segs = [np.expand_dims(np.array(PIL.Image.open(i)), axis=0) for i in seg_files_list[:647]]
segs0 = [np.expand_dims(np.full((np.array(PIL.Image.open(i)).shape[0], np.array(PIL.Image.open(i)).shape[1]), False), axis=0) for i in seg_files_list[647:]] 
segs.extend(segs0)

# shape 변형이 잘 되었는지 확인할 수 있다.
test_number = 660 
print(segsorigin[test_number].shape)
print(segs[test_number].shape)


# define transforms for image and segmentation
import torch
import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    AsChannelFirst,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureChannelFirst,
    AddChannel,
    RandRotate,
    RandFlip,
    RandZoom,
)
from monai.visualize import plot_2d_or_3d_image

from torch.utils.tensorboard import SummaryWriter

crop_size = 176 # UNet에서 maxpoolking을 4번 실행하기 때문에 crop_size를 16의 배수로 실행한다. --> 데이터 손실이 발생하므로 crop 말고 resize로 해보자.

train_imtrans = Compose(
    [
        #LoadImage(image_only=True, ensure_channel_first=True), # numpy 배열 그대로 넣어주기 때문에 load하지 않는다.
        #EnsureChannelFirst(),
        AsChannelFirst(), # shape가 (, , channel)이다. MONAI framework의 transform에서는 (channel, , ) shape를 가정하므로 변환한다.
        ScaleIntensity(),
        RandSpatialCrop((crop_size, crop_size), random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    ]
)
train_segtrans = Compose(
    [
        #LoadImage(image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        RandSpatialCrop((crop_size, crop_size), random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 1)),
    ]
) 


ds = ArrayDataset(segs, train_segtrans) 
print(type(ds))
for i in ds[640:]:
    print(i.shape)

val_imtrans = Compose([AsChannelFirst(), ScaleIntensity()])
val_segtrans = Compose([ScaleIntensity()]) #  validation에서는 crop, rotate를 하지 않는다. --> Check

test_imtrans = Compose([AsChannelFirst(), ScaleIntensity()])
test_segtrans = Compose([ScaleIntensity()]) #  test에서는 crop, rotate를 하지 않는다. --> Check

# define array dataset, data loader
check_ds = ArrayDataset(images, train_imtrans, segs, train_segtrans)
check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im, seg = monai.utils.misc.first(check_loader)
print(im.shape, seg.shape)

# shuffle
length = len(images)
indices = np.arange(length)
np.random.shuffle(indices)
images = [images[i] for i in indices] # shuffled list
segs = [segs[i] for i in indices] # shuffled list

val_frac = 0.15
test_frac = 0.15
test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split

# create a training data loader 
train_ds = ArrayDataset(images[val_split:], train_imtrans, segs[val_split:], train_segtrans)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
# create a validation data loader
val_ds = ArrayDataset(images[test_split:val_split], val_imtrans, segs[test_split:val_split], val_segtrans)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
# create a test data loader
test_ds = ArrayDataset(images[:test_split], test_imtrans, segs[:test_split], test_segtrans)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

print(f"Training count: {len(train_ds)}, Validation count: {len(val_ds)}, Test count: {len(test_ds)}")

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# create UNet, DiceLoss and Adam optimizer
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 0 to use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

print(len(train_ds))

model = monai.networks.nets.UNet(
    spatial_dims=2, 
    in_channels=3, # .png image의 channel이 3이다. --> ultrasound image가 흑백이기 때문에 channel=1로 변형해야 할 것으로 보인다.
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = monai.losses.DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()

print(len(train_ds))

for epoch in range(10):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{10}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        #print(inputs.shape)
        #print(labels.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(labels.shape)
        #print(outputs.shape)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                roi_size = (crop_size, crop_size)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels) # y_pred and y should have same shapes, got torch.Size([1, 1, 476, 560]) and torch.Size([1, 560, 1, 476]).
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_segmentation2d_array.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar("val_mean_dice", metric, epoch + 1)
            # plot the last model output as GIF image in TensorBoard with the corresponding image and label
            plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()


