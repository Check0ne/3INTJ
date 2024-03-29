from important.encoder.encoder import UNetEncoder

from typing import Optional, Union, List

from important.decoder.seg_decoder import UNetDecoder
from important.decoder.reg_decoder import AE_Decoder
#from .rec_decoder import AE_Decoder

#from ..encoders import get_encoder
from important.base.heads import SegmentationHead, ClassificationHead, ReconstructionHead
from important.base.model import Multi_Task_Model, Dual_Task_Model_CLS_SEG

import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

 ############ UP ############

## Smart-Net
class Up_SMART_Net(Multi_Task_Model):
    def __init__(
        self,
        encoder_name: str = "UNet",
    ):
        super().__init__()
        
        self.encoder = UNetEncoder()

        # CLS
        self.classification_head = ClassificationHead(
            in_channels=1024, 
            pooling='avg', 
            dropout=0.5,
            out_channels=1, 
        )

        # SEG
        self.seg_decoder = UNetDecoder()
        
        self.segmentation_head = SegmentationHead( # head 지금 사용 안 함.
            in_channels=64,
            out_channels=1,
            kernel_size=3,
        )

        # REC
        self.rec_decoder = AE_Decoder()
        
        self.reconstruction_head = ReconstructionHead(
            in_channels=64,
            out_channels=1,
            kernel_size=3,
        )

        self.name = "SMART-Net-{}".format(encoder_name)
        self.initialize()

## Dual
    # CLS+SEG
class Up_SMART_Net_Dual_CLS_SEG(Dual_Task_Model_CLS_SEG):
    def __init__(
        self,
        encoder_name: str = "UNet",
    ):
        super().__init__()

        self.encoder = UNetEncoder()

        # CLS
        self.classification_head = ClassificationHead(
            in_channels=1024, 
            pooling='avg', 
            dropout=0.5,
            out_channels=1, 
        )

        # SEG
        self.seg_decoder = UNetDecoder()
        
        self.segmentation_head = SegmentationHead( # head 지금 사용 안 함.
            in_channels=64,
            out_channels=1,
            kernel_size=3,
        )
        
        self.segmentation_head = SegmentationHead( # head 지금 사용 안 함.
            in_channels=64,
            out_channels=1,
            kernel_size=3,
        )

        self.name = "Dual-Net-{}".format(encoder_name)
        self.initialize()

############ DOWN ############

## DownTask - SMART-Net
class Down_SMART_Net_CLS(torch.nn.Module):
    def __init__(self):
        super(Down_SMART_Net_CLS, self).__init__()        
        
        self.encoder = Up_SMART_Net().encoder
        self.head = Up_SMART_Net().classification_head
        
        '''
        # Slicewise Feature Extract
        self.encoder = Up_SMART_Net().encoder
        self.pool    = torch.nn.AdaptiveAvgPool2d(1)
        
        # 3D Classifier - ResNet50 based
        self.LSTM    = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc      = nn.Linear(512*2, 512, True)
        self.relu    = nn.ReLU(True)
        self.drop    = nn.Dropout(p=0.5)
        
        # Head
        self.head    = nn.Linear(512, 1, True)   
        '''

    # # [fast version ... but causing OOM]
    # def forward(self, x, depths):

    #     B, C, H, W, D = x.shape

    #     # D stacking Features
    #     x = x.permute(4, 0, 1, 2, 3)                      # (B, C, H, W, D)   --> (D, B, C, H, W)
    #     x = torch.reshape(x, [D*B, C, H, W])              # (D, B, C, H, W)   --> (D*B, C, H, W)         
    #     x = self.encoder(x)[-1]                           # (D*B, C, H, W)    --> (D*B, C*, H*, W*)
    #     x = self.pool(x)                                  # (D*B, C*, H*, W*) --> (D*B, C*, 1, 1)  
    #     x = x.flatten(start_dim=1, end_dim=-1)            # (D*B, C*, 1, 1)   --> (D*B, C*)           
    #     x = torch.reshape(x, [D, B, x.shape[-1]])         # (D*B, C*)         --> (D, B, C*)
    #     x = x.permute(1, 0, 2)                            # (D, B, C*)        --> (B, D, C*)

    #     # 3D Classifier, Input = (Batch, Seq, Feat)
    #     self.LSTM.flatten_parameters()  
    #     x_packed = pack_padded_sequence(x, depths.cpu(), batch_first=True, enforce_sorted=False)
    #     RNN_out, (h_n, h_c) = self.LSTM(x_packed, None)    
    #     fc_input = torch.cat([h_n[-1], h_n[-2]], dim=-1) # Due to the Bi-directional
    #     x = self.fc(fc_input)
    #     x = self.relu(x)  
    #     x = self.drop(x)  
        
    #     # Head
    #     x = self.head(x)     

    #     return x
    
    def forward(self, x):
        
        feature_list = self.encoder(x)
        x = self.head(feature_list[-1])
        
        return x

    '''
    def forward(self, x, depths):
        
        # D stacking Features  
        encoder_embed_seq = []
        for i in range(x.shape[-1]):
            out = self.encoder(x[..., i])[-1]
            out = self.pool(out)
            out = out.view(out.shape[0], -1)
            encoder_embed_seq.append(out)   
            
        stacked_feat = torch.stack(encoder_embed_seq, dim=1)

        # 3D Classifier, Input = (Batch, Seq, Feat)
        self.LSTM.flatten_parameters()  
        x_packed = pack_padded_sequence(stacked_feat, depths.cpu(), batch_first=True, enforce_sorted=False)
        RNN_out, (h_n, h_c) = self.LSTM(x_packed, None)    
        fc_input = torch.cat([h_n[-1], h_n[-2]], dim=-1) # Due to the Bi-directional
        x = self.fc(fc_input)
        x = self.relu(x)  
        x = self.drop(x)  
        
        # Head
        x = self.head(x)     

        return x    
    '''
    


class Down_SMART_Net_SEG(torch.nn.Module):
    def __init__(self):
        super(Down_SMART_Net_SEG, self).__init__()

        # Slicewise Feature Extract
        self.encoder     = Up_SMART_Net().encoder
        self.seg_decoder = Up_SMART_Net().seg_decoder
        
        '''
        # 3D Segmentor - ResNet50 based
        self.conv1 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1   = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2   = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU()
        
        # Head
        self.head  = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        '''
        

    # # [fast version ... but causing OOM]
    # def forward(self, x):
    #     B, C, H, W, D = x.shape

    #     # D stacking Features
    #     x = x.permute(4, 0, 1, 2, 3)              # (B, C, H, W, D) --> (D, B, C, H, W)
    #     x = torch.reshape(x, [D*B, C, H, W])      # (D, B, C, H, W) --> (D*B, C, H, W)
    #     feat_list = self.encoder(x)               
    #     x         = self.seg_decoder(*feat_list)  

    #     # 3D Segmentor, Input = (D*B, C, H, W)
    #     x = torch.reshape(x, [D, B] + list(x.shape)[1:])   # (D*B, C*, H, W)  --> (D, B, C*, H, W)
    #     x = x.permute(1, 2, 3, 4, 0)                       # (D, B, C*, H, W) --> (B, C*, H, W, D)
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu1(x)
    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = self.relu2(x)
        
    #     # Head
    #     x = self.head(x)
        
    #     return x
    
    def forward(self, x):
        
        feature_list = self.encoder(x)
        x = self.seg_decoder(*feature_list)
        
        return x

    
    '''
    def forward(self, x):

        # D stacking Features  
        encoder_embed_seq = []
        for i in range(x.shape[-1]):
            feat_list = self.encoder(x[..., i])               
            seg_out   = self.seg_decoder(*feat_list)  
            encoder_embed_seq.append(seg_out)   
            
        stacked_feat = torch.stack(encoder_embed_seq, dim=-1)  # (B, C*, H, W, D)
        
        # 3D Segmentor, Input = (B, C*, H, W, D)
        x = self.conv1(stacked_feat)  
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Head
        x = self.head(x)
        
        return x
    '''
    








