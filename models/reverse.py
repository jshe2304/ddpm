'''
Reverse Diffusion Process. 
Implemented with U-Net. 
'''

import torch
import torch.nn as nn
from . import unet

class ReverseDiffusion(nn.Module):
    '''
    U-Net model for reverse diffusion steps. 
    '''

    def __init__(self, T, time_embedding_dim=32, time_n=512):
        super().__init__()
        
        self.time_embedding_dim = time_embedding_dim
        
        self.time_embedding = self.get_time_embeddings(T, time_embedding_dim, time_n)
        
        # Encoder
        self.down1 = unet.DownBlock(3, 32, time_embedding_dim)
        self.down2 = unet.DownBlock(32, 64, time_embedding_dim)
        
        # Bottleneck
        self.bottleneck = unet.BottleneckBlock(64, 128, time_embedding_dim)
        
        # Decoder
        self.up1 = unet.UpBlock(128, 64, 64)
        self.up2 = unet.UpBlock(64, 32, 32)
        
        # RGB Map
        self.rgb = nn.Sequential(
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Conv2d(32, 3, kernel_size=1)
        )

    def forward(self, x, t):
        time_embedding = self.time_embedding[t]
        
        # Encode
        residual1, x = self.down1(x, time_embedding)
        residual2, x = self.down2(x, time_embedding)
        
        # Bottleneck
        x = self.bottleneck(x, time_embedding)
        
        # Decode
        x = self.up1(x, residual2)
        x = self.up2(x, residual1)
        
        # Map to RBG
        x = self.rgb(x)
        
        return x

    @staticmethod
    def get_time_embeddings(T, time_embedding_dim, n=10000):
        '''
        Create sinusoidal time embeddings table. 
        See "Attention is all you need"
        '''
        n_timesteps = T + 1

        embeddings = torch.empty(n_timesteps, time_embedding_dim)

        for t in range(n_timesteps):
            for i in range(time_embedding_dim // 2):
                omega = n ** (-2 * i / time_embedding_dim)
                embeddings[t, 2 * i] = omega * t
                embeddings[t, 2 * i + 1] = omega * t

        embeddings[:, ::2] = torch.sin(embeddings[:, ::2]) # Even Columns
        embeddings[:, 1::2] = torch.cos(embeddings[:, 1::2]) # Odd Columns
        
        return embeddings
