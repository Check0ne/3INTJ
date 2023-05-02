#%%
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
data_dir = '/workspace/Prepocessed Dataset'

class_names = ['benign', 'malignant']
num_class = len(class_names)

images_dir = [
    sorted(glob(os.path.join(data_dir, class_names[i], "*).png")))
    for i in range (num_class)
]

images_dir[0] = images_dir[0][227:] # for dataset downsampling

masks_dir = [
    sorted(glob(os.path.join(data_dir, class_names[i], "*mask.png"))) 
    for i in range (num_class)
]

masks_dir[0] = masks_dir[0][227:] # for dataset downsampling

num_each = [len(images_dir[i]) for i in range(num_class)] # class 별 image_files 개수 리스트
images_class = [[0 for i in range(num_each[0])], [1 for i in range(num_each[1])]]

print(f"Benign: {num_each[0]}, Malignant: {num_each[1]}")

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
    EnsureChannelFirst,
    RandRotate90,
    RandFlip,
    Rotate90,
    Flip,
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter

resize_h = 224
resize_w = 224

def int_to_tensor(x):
    return torch.tensor(x, dtype=float).reshape((1,))

train_imtrans = Compose(
        [   
            LoadImage(image_only=True, ensure_channel_first=True, dtype=np.float32), 
            ScaleIntensity(),
            Rotate90(spatial_axes=(0, 1)),
            Flip(spatial_axis=0),
            Resize((resize_h,resize_w), mode='area'),
            
            # data augmentation
        ]
    )

train_segtrans = Compose(
    [
        LoadImage(image_only=True, ensure_channel_first=True, dtype=np.float32),
        Rotate90(spatial_axes=(0, 1)),
        Flip(spatial_axis=0),
        Resize((resize_h,resize_w), mode='area'),
    ]
)

clstrans = Compose([
    Lambda(int_to_tensor)
])

val_imtrans = train_imtrans
val_segtrans = train_segtrans

test_imtrans = train_imtrans
test_segtrans = train_segtrans

length = 210
val_frac = 0.1
test_frac = 0.1
test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split

benign_images_dir = images_dir[0]
malignant_images_dir = images_dir[1]
benign_masks_dir = masks_dir[0]
malignant_masks_dir = masks_dir[1]
benign_image_class = images_class[0]
malignant_image_class = images_class[1]

train_images = benign_images_dir[val_split:] + malignant_images_dir[val_split:]
val_images = benign_images_dir[test_split:val_split] + malignant_images_dir[test_split:val_split]
test_images = benign_images_dir[:test_split] + malignant_images_dir[:test_split]

train_masks = benign_masks_dir[val_split:] + malignant_masks_dir[val_split:]
val_masks = benign_masks_dir[test_split:val_split] + malignant_masks_dir[test_split:val_split]
test_masks = benign_masks_dir[:test_split] + malignant_masks_dir[:test_split]

train_image_class = benign_image_class[val_split:] + malignant_image_class[val_split:]
val_image_class = benign_image_class[test_split:val_split] + malignant_image_class[test_split:val_split]
test_image_class = benign_image_class[:test_split] + malignant_image_class[:test_split]

######################################################################################################## 여기까지 최적화 2023.05.02

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
