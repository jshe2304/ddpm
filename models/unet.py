'''
Implementation of a U-Net components for DDPM. 
See reverse.py for composed U-Net model. 
'''

import torch
import torch.nn as nn

class DownBlock(nn.Module):
    '''
    U-Net Downsampling Convolutional Block. 
    '''
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(in_channels if in_channels % 8 else 8, in_channels), 
            #nn.ReLU(), 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.time_mlp = nn.Sequential(
            #nn.ReLU(), 
            nn.Linear(time_embedding_dim, out_channels), 
            nn.SiLU(), 
            nn.Linear(out_channels, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(16, out_channels), 
            #nn.ReLU(), 
            #nn.Dropout(0.1), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.SiLU()
        )
        
        self.downsample = nn.MaxPool2d(2)
        
    def forward(self, x, time_embedding):
        x = self.conv1(x)
        x += self.time_mlp(time_embedding).unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(x)
        
        return x, self.downsample(x)
    
class BottleneckBlock(nn.Module):
    '''
    U-Net Bottleneck Convolutional Block. 
    '''
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(16, in_channels), 
            #nn.ReLU(), 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, out_channels), 
            nn.SiLU(), 
            nn.Linear(out_channels, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(16, out_channels), 
            #nn.ReLU(), 
            #nn.Dropout(0.1), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.SiLU()
        )

    def forward(self, x, time_embedding):
        x = self.conv1(x)
        x += self.time_mlp(time_embedding).unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(x)
        
        return x

class UpBlock(nn.Module):
    '''
    U-Net Upsampling Convolutional Block. 
    '''
    def __init__(self, in_channels, residual_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        #self.upconv = nn.Sequential(
        #    nn.GroupNorm(16, in_channels), 
        #    nn.ReLU(), 
        #    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), 
        #)
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(16, out_channels + residual_channels), 
            #nn.ReLU(), 
            nn.Conv2d(out_channels + residual_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(16, out_channels), 
            #nn.ReLU(), 
            #nn.Dropout(0.1), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.SiLU()
        )
        
    def forward(self, x, residual):
        if x.dim() != residual.dim() != 4:
            raise Exception('Bad dimensions.')

        x = self.upconv(x)
        x = torch.cat((x, residual), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x