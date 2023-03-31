import torch
import torch.nn as nn
import torch.nn.functional as F
from important.base import modules as md

class UNetDecoder(nn.Module): # nn.Module class를 UNet에 상속한다.
    def __init__(
            self,
            encoder_channels, # [1, 64, 64, 128, 128, 256, 256, 512, 512, 1024]
            decoder_channels, # [1024, 512, 512, 256, 256, 128, 128, 64, 64]
            use_batchnorm=False,
            attention_type=None,
            center=False,
    ):
        super(UNetDecoder, self).__init__()
        encoder_channels = encoder_channels[1:] # [64, 64, 128, 128, 256, 256, 512, 512, 1024]
        encoder_channels = encoder_channels[::-1]  # [1024, 512, 512, 256, 256, 128, 128, 64, 64]
        
        # computing blocks input and output channels
        head_channels = encoder_channels[0] # 1024
        in_channels   = [head_channels] + list(decoder_channels[:-1]) # [1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64]
        out_channels  = decoder_channels # [1024, 512, 512, 256, 256, 128, 128, 64, 64]
        
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):

            cbr = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias), # bias
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace = True)
            )
            
            return cbr
        
        # Expansive path
        self.dec5_1 = CBR(in_channels=in_channels[0], out_channels=out_channels[0]) # channel

        self.unpool4 = nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=in_channels[2], kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR(in_channels=in_channels[1], out_channels=out_channels[1])
        self.dec4_1 = CBR(in_channels=in_channels[2], out_channels=out_channels[2])

        self.unpool3 = nn.ConvTranspose2d(in_channels=out_channels[2], out_channels=in_channels[4], kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR(in_channels=in_channels[3], out_channels=out_channels[3])
        self.dec3_1 = CBR(in_channels=in_channels[4], out_channels=out_channels[4])

        self.unpool2 = nn.ConvTranspose2d(in_channels=out_channels[4], out_channels=in_channels[6], kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR(in_channels=in_channels[5], out_channels=out_channels[5])
        self.dec2_1 = CBR(in_channels=in_channels[6], out_channels=out_channels[6])

        self.unpool1 = nn.ConvTranspose2d(in_channels=out_channels[6], out_channels=in_channels[8], kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR(in_channels=in_channels[7], out_channels=out_channels[7])
        self.dec1_1 = CBR(in_channels=in_channels[8], out_channels=out_channels[8])

        #self.fc = nn.Conv2d(in_channels=in_channels[9], out_channels=out_channels[9], kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, *features):
        
        features = features[::-1]
        
        head = features[0]
        skips = features[1:] ## feature를 저장해 놓는 무언가가..
        
        dec5_1 = self.dec5_1(head)
        
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, skips[0]), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, skips[1]), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, skips[2]), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, skips[3]), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        x = dec1_1
        
        return x