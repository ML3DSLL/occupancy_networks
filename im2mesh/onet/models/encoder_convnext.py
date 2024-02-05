from .convnext import ConvNeXt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    ''' Latent encoder class.

    It encodes the input points and returns mean and standard deviation for the
    posterior Gaussian distribution.

    Args:
        z_dim (int): dimension of output code z
        c_dim (int): dimension of latent conditioned code c
        dim (int): input dimension
        leaky (bool): whether to use leaky ReLUs
    '''
    def __init__(self, z_dim=128, c_dim=128, dim=3, leaky=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3,3,kernel_size=4,stride=2)
        self.conv2 = nn.Conv2d(3,3,kernel_size=4,stride=2)
        self.convnext1 = ConvNeXt(in_chans=dim)
        self.convnext2 = ConvNeXt(in_chans=dim)
        self.convnext3 = ConvNeXt(in_chans=dim)
        self.fc_mean = nn.Linear(393, z_dim)
        self.fc_logstd = nn.Linear(393, z_dim)
    
    def forward(self, x, pose):
        #to be deleted
        x = F.interpolate(x, size = (128,128), mode = 'bilinear', align_corners=False)

        latent1 = self.convnext1(x)
        x = self.conv1(x)
        cache1 = x
        latent2 = self.convnext1(cache1)
        x = self.conv2(x)
        cache2 = x
        latent3 = self.convnext1(cache2)
        result = torch.concat(latent1,latent2)
        result = torch.concat(result,latent3)
        #input_pose_dim.shape = (9,1)
        result = torch.concat(result, pose)

        mean = self.fc_mean(result)
        logstd = self.fc_logstd(result)

        return mean, logstd
        
