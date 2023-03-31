import torch
import torch.nn as nn

class UNetEncoder(nn.Module): # nn.Module class를 UNet에 상속한다.
    def __init__(self, bilinear=False): ###
        super(UNetEncoder, self).__init__()
        
        self.in_channels = 1
        self.out_channels = [1, 64, 64, 128, 128, 256, 256, 512, 512, 1024] 
        self.bilinear = bilinear 
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            
            cbr = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias), 
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace = True)
            )
            
            return cbr
        
        # Contracting path
        self.enc1_1 = CBR(in_channels=self.in_channels, out_channels=self.out_channels[1])
        self.enc1_2 = CBR(in_channels=self.out_channels[1], out_channels=self.out_channels[2])
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.enc2_1 = CBR(in_channels=self.out_channels[2], out_channels=self.out_channels[3])
        self.enc2_2 = CBR(in_channels=self.out_channels[3], out_channels=self.out_channels[4])
        
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.enc3_1 = CBR(in_channels=self.out_channels[4], out_channels=self.out_channels[5])
        self.enc3_2 = CBR(in_channels=self.out_channels[5], out_channels=self.out_channels[6])
        
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4_1 = CBR(in_channels=self.out_channels[6], out_channels=self.out_channels[7]) 
        self.enc4_2 = CBR(in_channels=self.out_channels[7], out_channels=self.out_channels[8])
        
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.enc5_1 = CBR(in_channels=self.out_channels[8], out_channels=self.out_channels[9])
    
    def forward(self, x):
        
        features = []
        
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        features.append(enc1_2) # feature를 나중에 skip connection에 활용한다.
        pool1 = self.pool1(enc1_2)
        
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        features.append(enc2_2)
        pool2 = self.pool2(enc2_2)
        
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        features.append(enc3_2)
        pool3 = self.pool3(enc3_2)
        
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        features.append(enc4_2)
        pool4 = self.pool4(enc4_2)
      
        x = self.enc5_1(pool4)
        features.append(x)
    
        return features # feature 전체를 return.