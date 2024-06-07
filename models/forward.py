'''
Forward Diffusion Process. 
'''

import torch
import torch.nn as nn

class ForwardDiffusion(nn.Module):
    '''
    Forward Diffusion Process
    '''
    schedules = ['linear']
    
    def __init__(self, noise_param, T, schedule, device):
        super().__init__()
        
        self.T = T
        self.device = device
        
        self.betas = self.get_betas(noise_param, T, schedule).to(device)
        
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def step(self, x_i, i, j):
        '''
        Sample from q(x_j | x_i). 
        Batched inputs unsupported. 
        '''
        if i > j or i < 0 or j > self.T:
            raise Exception('Invalid step.')
        
        # q(x_i | x_i)
        if i == j: 
            return x_i, None
        # q(x_j | x_{j-1})
        elif i == j - 1: 
            mean = torch.sqrt(1 - self.betas[j]) * x_i
            std = torch.sqrt(self.betas[j])
        # q(x_j | x_0)
        elif i == 0 and j > 0: 
            mean = torch.sqrt(self.alpha_bars[j]) * x_i
            std = torch.sqrt(1 - self.alpha_bars[j])
        else:
            raise Exception('Not supported.')
        
        eps = torch.randn_like(x_i, device=self.device)
        
        return mean + std * eps, eps
    
    def forward(self, x_0, t):
        '''
        Sample from q(x_0 | x_t). 
        Batched inputs supported. 
        '''
        if type(t) is int:
            if t == 0: return x_0, None
            if 0 > t > self.t: 
                raise Exception('Invalid t.')
        elif torch.any(t < 0) or torch.any(t > self.T):
            raise Exception('Invalid t.')
        
        mean = torch.sqrt(self.alpha_bars[t]) * x_0
        std = torch.sqrt(1 - self.alpha_bars[t])
        eps = torch.randn_like(x_0, device=self.device)
        
        return mean + std * eps, eps
    
    @staticmethod
    def get_betas(noise_param, T, schedule):
        '''
        Create betas schedule Tensor
        '''
        if schedule == 'linear':
            if hasattr(noise_param, '__iter__') and len(noise_param) != 2:
                raise Exception('Bad noise parameter. Need 2-iterable.')
            if noise_param[0] <= 0 or noise_param[1] <= 0:
                raise Exception('Betas must be positive. ')
            
            return torch.cat((
                torch.Tensor([0]), 
                torch.linspace(*noise_param, T)
            ))
