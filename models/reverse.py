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

    def __init__(self, forward_process, time_embedding_dim=32, time_n=512, device=torch.device('cpu')):
        super().__init__()
        
        self.betas = forward_process.betas
        self.alphas = forward_process.alphas
        self.alpha_bars = forward_process.alpha_bars
        
        self.T = forward_process.T
        self.time_embedding_dim = time_embedding_dim
        self.device = device
        
        self.time_embeddings = self.get_time_embeddings(self.T, time_embedding_dim, time_n).to(device)
        
        # Encoder
        self.down1 = unet.DownBlock(3, 16, time_embedding_dim)
        self.down2 = unet.DownBlock(16, 32, time_embedding_dim)
        self.down3 = unet.DownBlock(32, 64, time_embedding_dim)

        # Bottleneck
        self.bottleneck = unet.BottleneckBlock(64, 128, time_embedding_dim)
        
        # Decoder
        self.up1 = unet.UpBlock(128, 64, 64)
        self.up2 = unet.UpBlock(64, 32, 32)
        self.up3 = unet.UpBlock(32, 16, 16)
        
        # RGB Map
        self.rgb = nn.Sequential(
            nn.GroupNorm(16, 16), 
            nn.ReLU(), 
            nn.Conv2d(16, 3, kernel_size=1)
        )

        self.to(device)

    def step(self, x, t):
        time_embedding = self.time_embeddings[t]
        
        # Encode
        residual1, x = self.down1(x, time_embedding)
        residual2, x = self.down2(x, time_embedding)
        residual3, x = self.down3(x, time_embedding)

        # Bottleneck
        x = self.bottleneck(x, time_embedding)
        
        # Decode
        x = self.up1(x, residual3)
        x = self.up2(x, residual2)
        x = self.up3(x, residual1)
        
        # Map to RBG
        x = self.rgb(x)
        
        return x
    
    def forward(self, x, record_steps=False):
        if record_steps:
            steps = [x.detach().cpu()]
        
        for t in range(self.T, 0, -1):
            z = torch.randn_like(x) if t > 0 else torch.zeros(x.shape)
            z = z.to(self.device)
            
            x -= (1-self.alphas[t]) * self.step(x, t) * ((1 - self.alpha_bars[t]) ** -0.5)
            x *= self.alphas[t] ** -0.5
            x += z * (self.betas[t] ** 0.5)

            #x = torch.clamp(x, -1, 1)
            x /= torch.max(x.max(), torch.abs(x.min()))

            if record_steps:
                steps.append(x.detach().cpu())
            
        return steps if record_steps else x

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
