#%%
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import json
import time
import random
import datetime
import argparse
from pathlib import Path
from PIL import Image
from glob import glob
from model import *

from engine import *
from losses import Uptask_Loss, Downtask_Loss
from optimizers import create_optim
from lr_schedulers import create_scheduler

from monai.data import ArrayDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
    SaveImage,
    Resize,
    Lambda,
    Rotate90,
    Flip,
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('SMART-Net Framework Train and Test script', add_help=False)

    # Setting Method
    parser.add_argument('--method', default='Single', choices=['Multi', 'Single'], type=str, help='Learning method')
    
    # Setting Upstream, Downstream, task
    parser.add_argument('--training-stream', default='Downstream', choices=['Upstream', 'Downstream'], type=str, help='training stream') 
    parser.add_argument('--task', default='SEG', choices=['CLS', 'SEG'], type=str, help='task(CLS/SEG)')
    
    # Model parameters
    parser.add_argument('--model-name', default='Down_SMART_Net_SEG', choices=['Up_SMART_Net', 'Up_SMART_Net_Dual_CLS_SEG','Down_SMART_Net_CLS', 'Down_SMART_Net_SEG'], type=str, help='training stream') 
    
    # Learning rate and schedule and Epoch parameters
    parser.add_argument('--epochs', default=100, type=int, help='Upstream 1000 epochs, Downstream 500 epochs')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')  
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    
    # Option
    parser.add_argument('--gradual-unfreeze', type=str2bool, default="TRUE", help='gradual unfreezing the encoder for Downstream Task')
    
    # Continue Training
    parser.add_argument('--resume',           default='',  help='resume from checkpoint')  # '' = None
    parser.add_argument('--from-pretrained',  default='',  help='pre-trained from checkpoint')
    parser.add_argument('--load-weight-type', default='',  help='the types of loading the pre-trained weights')
    
    # DataLoader setting
    parser.add_argument('--batch-size',  default=20, type=int)
    parser.add_argument('--num-workers', default=10, type=int)
    
    # Optimizer parameters
    parser.add_argument('--optimizer', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    
    # Validation setting
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    
    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode',       default='Single', choices=['DataParallel', 'Single'], type=str, help='multi-gpu-mode') 
    parser.add_argument('--cuda-visible-devices', default='0', type=str, help='cuda_visible_devices')
    
    
    # Prediction and Save setting
    parser.add_argument('--output-dir', default='/workspace/trained_model/single/seg', help='path where to save, empty for no saving')
    
    return parser

# Fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def main(args):
    
     # data load
    data_dir = '/workspace/BUS_data'
    train_images = sorted(glob(os.path.join(data_dir, "train/input", "*.png")))
    train_masks = sorted(glob(os.path.join(data_dir, "train/seg_label", "*.png")))
    train_image_class = [0 if 'benign' in i else 1 for i in train_images]
    val_images = sorted(glob(os.path.join(data_dir, "val/input", "*.png")))
    val_masks = sorted(glob(os.path.join(data_dir, "val/seg_label", "*.png")))
    val_image_class = [0 if 'benign' in i else 1 for i in val_images]
    
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

    # create a training data loader 
    train_ds = ArrayDataset(train_images, train_imtrans, train_masks, train_segtrans, train_image_class, clstrans)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = ArrayDataset(val_images, val_imtrans, val_masks, val_segtrans, val_image_class, clstrans)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # create a test data loader
    #test_ds = ArrayDataset(test_images, test_imtrans, test_masks, test_segtrans, test_image_class, clstrans)
    #test_loader = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    print(f"Training count: {len(train_ds)}, Validation count: {len(val_ds)}")
    #print(f"Training count: {len(train_ds)}, Validation count: {len(val_ds)}, Test count: {len(test_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    # Select Model
    if args.method == 'Multi':
        if args.training_stream == 'Upstream':
            if args.model_name == 'Up_SMART_Net': 
                model = Up_SMART_Net()
                criterion = Uptask_Loss(name=args.model_name)
            else :
                model = Up_SMART_Net_Dual_CLS_SEG()
                criterion = Uptask_Loss(name=args.model_name)
        else :
            if args.task == 'CLS':
                model = Down_SMART_Net_CLS()
                criterion = Downtask_Loss(name=args.model_name)
            else:
                model = Down_SMART_Net_SEG()
                criterion = Downtask_Loss(name=args.model_name)
    else:
        if args.task == 'CLS':
            model = Down_SMART_Net_CLS()
            criterion = Downtask_Loss(name=args.model_name)
        else:
            model = Down_SMART_Net_SEG()
            criterion = Downtask_Loss(name=args.model_name)        
            
    # Optimizer 
    optimizer = create_optim(name=args.optimizer, model=model)
    
    global log_dict
    
    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])        
        optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])        
        args.start_epoch = checkpoint['epoch'] + 1 
        try:
            log_path = os.path.dirname(args.resume)+'/log.txt'
            lines    = open(log_path,'r').readlines()
            val_loss_list = []
            for l in lines:
                log_dict = json.loads(l.replace('NaN', '0'))
                val_loss_list.append(log_dict['valid_loss'])
            print("Epoch: ", np.argmin(val_loss_list), " Minimum Val Loss ==> ", np.min(val_loss_list))
        except:
            pass
        
        # Optimizer Error fix...!
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    
    # Using the pre-trained feature extract's weights
    if args.from_pretrained:
        print("Loading... Pre-trained")      
        model_dict = model.state_dict() 
        print("Check Before weight = ", model_dict['encoder.enc1_1.0.weight'].std().item())
        checkpoint = torch.load(args.from_pretrained, map_location='cpu')
        if args.load_weight_type == 'full':
            model.load_state_dict(checkpoint['model_state_dict'])   
        elif args.load_weight_type == 'encoder':
            filtered_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if (k in model_dict) and ('encoder.' in k)}
            model_dict.update(filtered_dict)             
            model.load_state_dict(model_dict)   
        print("Check After weight  = ", model.state_dict()['encoder.enc1_1.0.weight'].std().item())        
        
    data_loader_train = train_loader
    data_loader_valid = val_loader

    # Multi GPU
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)
        model.to(device)
    elif args.multi_gpu_mode == 'Single':
        model.to(device)
    else :
        raise Exception('Error...! args.multi_gpu_mode') 

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_valid_auc = 0
    best_valid_dice = 0
    best_auc_epoch = 0
    best_dice_epoch = 0
    
    # Whole LOOP
    for epoch in range(args.start_epoch, args.epochs): 
        
        if args.method == 'Multi':
            if args.training_stream == 'Upstream':
                if args.model_name == 'Up_SMART_Net':
                    train_stats = train_Up_SMART_Net(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                    print("Averaged train_stats: ", train_stats)
                    valid_stats = valid_Up_SMART_Net(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
                    print("Averaged valid_stats: ", valid_stats)
                else :
                    train_stats = train_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                    print("Averaged train_stats: ", train_stats)
                    valid_stats = valid_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
                    print("Averaged valid_stats: ", valid_stats)
            elif args.training_stream == 'Downstream':
                if args.task == 'CLS':
                    train_stats = train_Down_SMART_Net_CLS(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size, args.gradual_unfreeze)
                    print("Averaged train_stats: ", train_stats)
                    valid_stats = valid_Down_SMART_Net_CLS(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
                    print("Averaged valid_stats: ", valid_stats)
                else :
                    train_stats = train_Down_SMART_Net_SEG(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size, args.gradual_unfreeze)
                    print("Averaged train_stats: ", train_stats)
                    valid_stats = valid_Down_SMART_Net_SEG(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
                    print("Averaged valid_stats: ", valid_stats)
        elif args.method == 'Single':
            if args.task == 'CLS':
                train_stats = train_Up_SMART_Net_Single_CLS(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                print("Averaged train_stats: ", train_stats)
                valid_stats = valid_Up_SMART_Net_Single_CLS(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
                print("Averaged valid_stats: ", valid_stats)
            else:
                train_stats = train_Up_SMART_Net_Single_SEG(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                print("Averaged train_stats: ", train_stats)
                valid_stats = valid_Up_SMART_Net_Single_SEG(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
                print("Averaged valid_stats: ", valid_stats)
        else :
            raise KeyError("Wrong training stream `{}`".format(args.training_stream))

    # Save & Prediction png
        if args.method == 'Multi' and args.training_stream == 'Upstream':
            if best_valid_auc < valid_stats['auc']:
                best_valid_auc = valid_stats['auc']
                best_auc_epoch = epoch
                best_auc_paths = args.output_dir + '_best_auc.pth'
                torch.save({
                'model_state_dict': model.state_dict() if args.multi_gpu_mode == 'Single' else model.module.state_dict(), # multi-gpu mode?
                'optimizer': optimizer.state_dict(),
                #'lr_scheduler': lr_scheduler.state_dict(), # scheduler 역할
                'epoch': epoch,
                'args': args,
                }, best_auc_paths)

            if best_valid_dice < valid_stats['dice']:
                best_valid_dice = valid_stats['dice']
                best_dice_epoch = epoch
                best_dice_paths = args.output_dir + '_best_dice.pth'
                torch.save({
                'model_state_dict': model.state_dict() if args.multi_gpu_mode == 'Single' else model.module.state_dict(), # multi-gpu mode?
                'optimizer': optimizer.state_dict(),
                #'lr_scheduler': lr_scheduler.state_dict(), # scheduler 역할
                'epoch': epoch,
                'args': args,
                }, best_dice_paths)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'valid_{k}': v for k, v in valid_stats.items()},
                        'epoch': epoch, 'best_auc_epoch': best_auc_epoch,'best_dice_epoch': best_dice_epoch}
        else:
            if args.task == 'CLS':
                if best_valid_auc < valid_stats['auc']:
                    best_valid_auc = valid_stats['auc']
                    best_auc_epoch = epoch
                    best_auc_paths = args.output_dir + '_best_auc.pth'
                    torch.save({
                    'model_state_dict': model.state_dict() if args.multi_gpu_mode == 'Single' else model.module.state_dict(), # multi-gpu mode?
                    'optimizer': optimizer.state_dict(),
                    #'lr_scheduler': lr_scheduler.state_dict(), # scheduler 역할
                    'epoch': epoch,
                    'args': args,
                    }, best_auc_paths)
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'valid_{k}': v for k, v in valid_stats.items()},
                        'epoch': epoch, 'best_auc_epoch': best_auc_epoch}
            else:
                if best_valid_dice < valid_stats['dice']:
                    best_valid_dice = valid_stats['dice']
                    best_dice_epoch = epoch
                    best_dice_paths = args.output_dir + '_best_dice.pth'
                    torch.save({
                    'model_state_dict': model.state_dict() if args.multi_gpu_mode == 'Single' else model.module.state_dict(), # multi-gpu mode?
                    'optimizer': optimizer.state_dict(),
                    #'lr_scheduler': lr_scheduler.state_dict(), # scheduler 역할
                    'epoch': epoch,
                    'args': args,
                    }, best_dice_paths)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'valid_{k}': v for k, v in valid_stats.items()},
                            'epoch': epoch,'best_dice_epoch': best_dice_epoch}    
        if args.output_dir:
            with open(args.output_dir + "/log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        #lr_scheduler.step(epoch)

    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SMART-Net Framework training and evaluation script', parents=[get_args_parser()])
    args   = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    os.environ["CUDA_VISIBLE_DEVICES"]  =  args.cuda_visible_devices        
    
    main(args)
