from important.encoder.encoder import UNetEncoder

from typing import Optional, Union, List

from important.decoder.seg_decoder import UNetDecoder
from important.decoder.reg_decoder import AE_Decoder
#from .rec_decoder import AE_Decoder

#from ..encoders import get_encoder
from important.base.heads import SegmentationHead, ClassificationHead, ReconstructionHead
from important.base.model import Multi_Task_Model

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


'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 0 to use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Up_SMART_Net().to(device) 
test_input = torch.rand(3, 1, 224, 224).to(device)
print(test_input.shape)
test_output = model.forward(test_input)
print(test_output[0].shape)
print(test_output[1].shape)
print(test_output[2].shape)

'''








