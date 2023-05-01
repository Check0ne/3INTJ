import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import json
import time
import datetime
from PIL import Image
from glob import glob
from model import *

from engine import *
from losses import Uptask_Loss, Downtask_Loss
from optimizers import create_optim
from lr_schedulers import create_scheduler

# data load
data_dir = '/workspace/Dataset_BUSI/Dataset_BUSI_with_GT'

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
    image_class.extend([np.array(i)] * num_each[i]) # class 이름 리스트 생성 [A, A, A, ... B, B, B, ... C, C, C ...] class 요소의 개수는 num_each 
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


images = images[227:] # for downsampling
new_segs= new_segs[227:] # for downsampling
image_class = image_class[227:] # for downsampling
print(f"Case counts after down sampling: {len(images)}")


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
    Lambda,
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


def int_to_tensor(x):
    return torch.tensor(x, dtype=float).reshape((1,))


clstrans = Compose([
    Lambda(int_to_tensor)
])


print(images[0].shape)
ds = ArrayDataset(images, train_imtrans) 
print(type(ds))
print(ds[0].shape) # channel이 맨 앞에 있는 tensor로 변환.

val_imtrans = Compose([AsChannelFirst(), Resize((resize_h, resize_w), mode='area'), ScaleIntensity()])
val_segtrans = Compose([AsChannelFirst(), Resize((resize_h, resize_w), mode='area'), ScaleIntensity()]) #  validation에서는 crop, rotate를 하지 않는다. --> Check

test_imtrans = Compose([AsChannelFirst(), Resize((resize_h, resize_w), mode='area'), ScaleIntensity()])
test_orgtrans = Compose([AsChannelFirst(), ScaleIntensity()])
test_segtrans = Compose([AsChannelFirst(), ScaleIntensity()]) #  test에서는 crop, rotate를 하지 않는다. --> Check

# define array dataset, data loader
check_ds = ArrayDataset(images, train_imtrans, new_segs, train_segtrans, image_class, clstrans) 
check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im, seg, cls = monai.utils.misc.first(check_loader)
print(im.shape, seg.shape, cls.shape)

'''

'''

images_b = images[:210] # for downsampling
images_m = images[210:] # for downsampling
new_segs_b = new_segs[:210] # for downsampling
new_segs_m = new_segs[210:] # for downsampling
image_class_b = image_class[:210] # for downsampling
image_class_m = image_class[210:] # for downsampling

length = int(len(images)*0.5)

# shuffle benign
indices = np.arange(length)
np.random.shuffle(indices)
images_b = [images_b[i] for i in indices] # shuffled list
new_segs_b = [new_segs_b[i] for i in indices] # shuffled list
image_class_b = [image_class_b[i] for i in indices] # shuffeld list

# shuffle malignant
np.random.shuffle(indices)
images_m = [images_m[i] for i in indices] # shuffled list
new_segs_m = [new_segs_m[i] for i in indices] # shuffled list
image_class_m = [image_class_m[i] for i in indices] # shuffeld list

val_frac = 0.1
test_frac = 0.1
test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split

train_images = images_b[val_split:] + images_m[val_split:]
val_images = images_b[test_split:val_split] + images_m[test_split:val_split]
test_images = images_b[:test_split] + images_m[:test_split]

train_new_segs = new_segs_b[val_split:] + new_segs_m[val_split:]
val_new_segs = new_segs_b[test_split:val_split] + new_segs_m[test_split:val_split]
test_new_segs = new_segs_b[:test_split] + new_segs_m[:test_split]

train_image_class = image_class_b[val_split:] + image_class_m[val_split:]
val_image_class = image_class_b[test_split:val_split] + image_class_m[test_split:val_split]
test_image_class = image_class_b[:test_split] + image_class_m[:test_split]

# create a training data loader 
train_ds = ArrayDataset(train_images, train_imtrans, train_new_segs, train_segtrans, train_image_class, clstrans)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

# create a validation data loader
val_ds = ArrayDataset(val_images, val_imtrans, val_new_segs, val_segtrans, val_image_class, clstrans)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

# create a test data loader
test_ds = ArrayDataset(test_images, test_imtrans, test_new_segs, test_segtrans, test_image_class, clstrans)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

print(f"Training count: {len(train_ds)}, Validation count: {len(val_ds)}, Test count: {len(test_ds)}")

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=False) # ignore_empty option!!!! normal case를 위해서 꼭 필요하다.
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")

# create Network, DiceLoss and Adam optimizer
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 0 to use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

print(len(train_ds)) #test



training_stream = input("Training Stream: ")

# Select Model
model_name = input("Model(Up_SMART_Net/Down_SMART_Net_CLS/Down_SMART_Net_SEG): ")
if model_name == 'Up_SMART_Net':
    model = Up_SMART_Net().to(device)
elif model_name == 'Down_SMART_Net_CLS':
    model = Down_SMART_Net_CLS().to(device)
    output_dir = '/workspace/trained_model/down/cls'
else:
    model = Down_SMART_Net_SEG().to(device)
    output_dir = '/workspace/trained_model/down/seg'
    
# Select Loss
if training_stream == 'Upstream':
        criterion = Uptask_Loss(name=model_name)
        output_dir = '/workspace/trained_model/up'
else :
    criterion = Downtask_Loss(name=model_name)
    

# Optimizer & LR Scheduler
optimizer_name = 'adam'
lr_scheduler_name = 'poly_lr'
optimizer = create_optim(name=optimizer_name, model=model)
#lr_scheduler = create_scheduler(name=lr_scheduler_name, optimizer=optimizer) 인자 설정...


start_epoch = 0

resume = input("Resume(T/F): ") # resume은 있는 파일에서 그대로 이어서.. downstream하려면 pre trained로 weight 불러와서 모델에 넣어야 할 듯...?
if resume == 'T':
    resume = input("Model dir: ")
    checkpoint = torch.load(resume, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])        
    optimizer.load_state_dict(checkpoint['optimizer'])
    #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])        
    start_epoch = checkpoint['epoch'] + 1  
    try:    
        log_path = os.path.dirname(resume)+'/log.txt'
        lines    = open(log_path,'r').readlines()
        val_loss_list = []
        for l in lines:
            exec('log_dict='+l.replace('NaN', '0'))
            val_loss_list.append(log_dict['valid_loss']) ##?
        print("Epoch: ", np.argmin(val_loss_list), " Minimum Val Loss ==> ", np.min(val_loss_list))
    except:
        pass

    # Optimizer Error fix...!
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

pt = input("Pre trained(T/F): ")

# Using the pre-trained feature extract's weights
if pt == 'T':
    from_pretrained = input("Pre trained dir: ")
    load_weight_type = input("Encoder type(full/encoder): ")
    print("Loading... Pre-trained")      
    model_dict = model.state_dict() 
    print("Check Before weight = ", model_dict['encoder.enc1_1.0.weight'].std().item())
    checkpoint = torch.load(from_pretrained, map_location='cpu')
    if load_weight_type  == 'full':
        model.load_state_dict(checkpoint['model_state_dict'])   
    elif load_weight_type  == 'encoder':
        filtered_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if (k in model_dict) and ('encoder.' in k)}
        model_dict.update(filtered_dict)             
        model.load_state_dict(model_dict)   
    print("Check After weight  = ", model.state_dict()['encoder.enc1_1.0.weight'].std().item())



batch_size = 4
print_freq = 21

data_loader_train = train_loader
data_loader_valid = val_loader


# Multi GPU
multi_gpu_mode = 'Single'

# Option
gradual_unfreeze = True

epochs = input("Epochs: ")
epochs = int(epochs)
print(f"Start training for {epochs} epochs")
start_time = time.time()

# Whole LOOP
for epoch in range(start_epoch, start_epoch+epochs): 
    # Train & Valid
    if training_stream == 'Upstream':
        if model_name == 'Up_SMART_Net':
            train_stats = train_Up_SMART_Net(model, criterion, data_loader_train, optimizer, device, epoch, print_freq, batch_size)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_Up_SMART_Net(model, criterion, data_loader_valid, device, print_freq, batch_size)
            print("Averaged valid_stats: ", valid_stats)
    elif training_stream == 'Downstream':
        if model_name == 'Down_SMART_Net_CLS':
            train_stats = train_Down_SMART_Net_CLS(model, criterion, data_loader_train, optimizer, device, epoch, print_freq, batch_size, gradual_unfreeze)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_Down_SMART_Net_CLS(model, criterion, data_loader_valid, device, print_freq, batch_size)
            print("Averaged valid_stats: ", valid_stats)
        elif model_name == 'Down_SMART_Net_SEG':
            train_stats = train_Down_SMART_Net_SEG(model, criterion, data_loader_train, optimizer, device, epoch, print_freq, batch_size, gradual_unfreeze)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_Down_SMART_Net_SEG(model, criterion, data_loader_valid, device, print_freq, batch_size)
            print("Averaged valid_stats: ", valid_stats)
        
    else :
        raise KeyError("Wrong training stream `{}`".format(training_stream))

# Up down folder 나누기 --> 일단 up model은 저장이 잘 된다.
# Save & Prediction png
    checkpoint_paths = output_dir + '/epoch_' + str(epoch) + '_checkpoint.pth'
    torch.save({
        'model_state_dict': model.state_dict() if multi_gpu_mode == 'Single' else model.module.state_dict(), # multi-gpu mode?
        'optimizer': optimizer.state_dict(),
        #'lr_scheduler': lr_scheduler.state_dict(), # scheduler 역할
        'epoch': epoch,
        #'args': args,
    }, checkpoint_paths)

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                **{f'valid_{k}': v for k, v in valid_stats.items()},
                'epoch': epoch}
    
    if output_dir:
        with open(output_dir + "/log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    #lr_scheduler.step(epoch)



# Finish
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
