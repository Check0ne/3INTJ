#%%
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from glob import glob
from model import *

# data load
data_dir = '/workspace/3INTJ/Dataset_BUSI/Dataset_BUSI_with_GT'

class_names = ['benign', 'malignant']
#class_names = ['benign', 'malignant', 'normal']
num_class = len(class_names)

image_files = [
    sorted(glob(os.path.join(data_dir, class_names[i], "*).png")))
    for i in range (num_class)
]

seg_files = [
    sorted(glob(os.path.join(data_dir, class_names[i], "*mask*.png"))) # mask가 여러 개인 image의 경우, 첫 번째 mask만 불러오고 있다. --> mask 병합이 필요하다.
    for i in range (num_class)
]

new_seg_files = [
    sorted(glob(os.path.join(data_dir, class_names[i], "*mask.png"))) # mask가 여러 개인 image의 경우, 첫 번째 mask만 불러오고 있다. --> mask 병합이 필요하다.
    for i in range (num_class)
]

# print(len(image_files[0]), len(image_files[1]), len(image_files[2]))
print(len(image_files[0]), len(image_files[1]))

num_each = [len(image_files[i]) for i in range(num_class)] # class 별 image_files 개수 리스트

image_files_list = [] 
seg_files_list = []
new_seg_files_list = []
image_class = [] # benign --> 0, malignant --> 1

for i in range(num_class): 
    image_files_list.extend(image_files[i]) # image_files[i] 리스트의 요소(image)를 image_files_list에 추가
    seg_files_list.extend(seg_files[i])
    new_seg_files_list.extend(new_seg_files[i])
    image_class.extend([i] * num_each[i]) # class 이름 리스트 생성 [A, A, A, ... B, B, B, ... C, C, C ...] class 요소의 개수는 num_each 
num_total = len(image_class) # image_class의 요소 개수
image_width, image_height = Image.open(image_files_list[0]).size # 하나의 이미지 열어서 width, height 값 추출

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")

# normal_mask만 numpy arry가 3차원이다. 기존의 channel=3을 없애고 다시 channel=1을 추가했다. transform까지 정상적으로 되는 것을 확인할 수 있다.
# 기존에는 MONAI에서 파일명으로 접근했지만, 배열을 수정해야 하기 때문에 파일을 열어서 배열을 수정하고, 수정한 배열을 리스트에 넣는다.
images = [np.expand_dims(np.array(PIL.Image.open(i))[:, :, 0], axis=2) for i in image_files_list]

segs = [] # mask numpy array before merging

for i in seg_files_list:
    if np.array(PIL.Image.open(i)).ndim == 2:
        segs.append(np.expand_dims(np.array(PIL.Image.open(i)), axis=2))
    else:
        segs.append(np.expand_dims(np.array(PIL.Image.open(i))[:, :, 0], axis=2))

print(len(segs)) # 798

new_segs = [] # mask numpy array after merging

for i in range(len(segs)):
    if i == 0:
        temp_dir = seg_files_list[i]
        temp_arr = segs[i]
    else:
        new_dir = seg_files_list[i]
        new_arr = segs[i]
        if temp_dir.replace('.png','') in new_dir:
            temp_arr = temp_arr | new_arr
        else:
            new_segs.append(temp_arr)
            temp_dir = new_dir
            temp_arr = new_arr
    if i == len(segs)-1:
        new_segs.append(temp_arr)              
            
print(len(new_segs)) # 780


m = 2
print(new_seg_files_list[m])
plt.imshow(new_segs[m]*255, cmap = "gray", vmin=0, vmax=255)
plt.tight_layout()
plt.show()

#%%
# new_seg_files_list는 segmentation mask 파일 이름으로 이루어진 list (총 780개)
# new_seg는 segmentation mask array로 이루어진 list (True, False, 총 780개, 병합 완료)

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
    ScaleIntensity,
    AddChannel,
    SaveImage,
    Resize,
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter

resize_h = 224 # UNet에서 maxpoolking을 4번 실행하기 때문에 crop_size를 16의 배수로 실행한다. --> 데이터 손실이 발생하므로 crop 말고 resize로 해보자.
resize_w = 224

train_imtrans = Compose(
    [
        #LoadImage(image_only=True, ensure_channel_first=True), # numpy 배열 그대로 넣어주기 때문에 load하지 않는다.
        #EnsureChannelFirst(),
        AsChannelFirst(), # shape가 (, , channel)이다. MONAI framework의 transform에서는 (channel, , ) shape를 가정하므로 변환한다.
        #RemoveRepeatedChannel(repeats=3), # 중복되는 channe=3을 channel=1로 줄인다.
        Resize((resize_h,resize_w), mode='area'),
        ScaleIntensity(),
    ]
)
train_segtrans = Compose(
    [
        #LoadImage(image_only=True, ensure_channel_first=True),
        AsChannelFirst(),
        Resize((resize_h,resize_w), mode='area'),
        ScaleIntensity(),
    ]
)

train_clstrans = Compose([AsDiscrete(to_onehot=num_class)])

print(images[0].shape)
ds = ArrayDataset(images, train_imtrans) 
print(type(ds))
print(ds[0].shape) # channel이 맨 앞에 있는 tensor로 변환.

val_imtrans = Compose([AsChannelFirst(), Resize((resize_h,resize_w), mode='area'), ScaleIntensity()])
val_segtrans = Compose([AsChannelFirst(), ScaleIntensity()]) #  validation에서는 crop, rotate를 하지 않는다. --> Check

test_imtrans = Compose([AsChannelFirst(), Resize((resize_h,resize_w), mode='area'), ScaleIntensity()])
test_orgtrans = Compose([AsChannelFirst(), ScaleIntensity()])
test_segtrans = Compose([AsChannelFirst(), ScaleIntensity()]) #  test에서는 crop, rotate를 하지 않는다. --> Check

# define array dataset, data loader
check_ds = ArrayDataset(images, train_imtrans, new_segs, train_segtrans) 
check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im, seg = monai.utils.misc.first(check_loader)
print(im.shape, seg.shape)

# shuffle
length = len(images)
indices = np.arange(length)
np.random.shuffle(indices)
images = [images[i] for i in indices] # shuffled list
new_segs = [new_segs[i] for i in indices] # shuffled list
image_class = [image_class[i] for i in indices]

#k = train_clstrans(image_class) # 0 --> [1, 0], 1 --> [1, 0]

val_frac = 0.15
test_frac = 0.15
test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split

# create a training data loader 
train_ds = ArrayDataset(images[val_split:], train_imtrans, new_segs[val_split:], train_segtrans)
train_loader = DataLoader(train_ds, batch_size=3, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
# create a validation data loader
val_ds = ArrayDataset(images[test_split:val_split], val_imtrans, new_segs[test_split:val_split], val_segtrans)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
# create a test data loader
test_ds = ArrayDataset(images[:test_split], test_imtrans, new_segs[:test_split], test_segtrans, images[:test_split], test_orgtrans)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

print(f"Training count: {len(train_ds)}, Validation count: {len(val_ds)}, Test count: {len(test_ds)}")

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=False) ############# ignore_empty option!!!! normal case를 위해서 꼭 필요하다.
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")
# create UNet, DiceLoss and Adam optimizer
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 0 to use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

print(len(train_ds)) ##test

model = UNet(1, 1, 2).to(device)
#print(model)
'''
model = monai.networks.nets.UNet(
    spatial_dims=2, 
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
'''

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

num_epoch = 100

for epoch in range(num_epoch):
    print("-" * num_epoch)
    print(f"epoch {epoch + 1}/{num_epoch}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
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
                roi_size = (resize_h,resize_w)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                
                original_h = val_labels.shape[2]
                original_w = val_labels.shape[3]
                val_outputs = F.interpolate(val_outputs, size=(original_h, original_w), mode='bilinear', align_corners=False)
                
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                
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
#%%
#test
import torchvision.transforms as T
model.load_state_dict(torch.load("best_metric_model_segmentation2d_array.pth"))
transform = T.ToPILImage()
model.eval()
with torch.no_grad():
    
    metric = 0
    for test_data in test_loader:
        test_images, test_labels, test_org = test_data[0].to(device), test_data[1].to(device), test_data[2].to(device)
        # define sliding window size and batch size for windows inference
        roi_size = (resize_h,resize_w)
        sw_batch_size = 4
        test_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
        
        original_h = test_labels.shape[2]
        original_w = test_labels.shape[3]
        test_outputs = F.interpolate(test_outputs, size=(original_h, original_w), mode='bilinear', align_corners=False)
        
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
        test_labels = decollate_batch(test_labels)
        # compute metric for current iteration
        dice_score = dice_metric(y_pred=test_outputs, y=test_labels)
        
        #saveimg = transform(test_org[0])
        #saveimg.save(f"{dice_score.item()}input.png")
        #saveimg = transform(test_outputs[0])
        #saveimg.save(f"{dice_score.item()}output.png")
        #arr = test_labels[0].cpu().numpy().astype('float32')
        #arr = (arr*255).astype('uint8')
        #saveimg = PIL.Image.fromarray(arr[0].squeeze(), mode = 'L')
        #saveimg.save(f"{dice_score.item()}label.png")
        
    # aggregate the final mean dice result
    metric = dice_metric.aggregate().item()
    print("evaluation metric:", metric)
    # reset the status
    dice_metric.reset()
#%%
