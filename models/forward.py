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
    
    def __init__(self, noise_param, T, schedule):
        super().__init__()
        
        self.T = T
        self.betas = self.get_betas(noise_param, T, schedule)
        
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
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
        
    def forward(self, x_i, i, j):
        '''
        Forward noise step from x_i to x_j. 
        Noise process given by q(x_j | x_i). 
        '''
        if i == j: return x_i
        if i > j or i < 0 or j > self.T:
            raise Exception('Invalid step. ')
        
        # q(x_j | x_{j-1})
        if i == j - 1: 
            mean = torch.sqrt(1 - self.betas[j]) * x_i
            std = torch.sqrt(self.betas[j])
        
        # q(x_j | x_0)
        elif i == 0 and j > 0: 
            mean = torch.sqrt(self.alpha_bars[j]) * x_i
            std = torch.sqrt(1 - self.alpha_bars[j])
            
        else:
            raise Exception('Not supported.')
        
        return mean + std * torch.randn_like(x_i)