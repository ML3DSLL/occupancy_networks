import convnext
import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self, input_channel = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3,3,kernel_size=4,stride=2)
        self.conv2 = nn.Conv2d(3,3,kernel_size=4,stride=2)
        self.convnext1 = convnext.ConvNeXt(in_chans=input_channel)
        self.convnext2 = convnext.ConvNeXt(in_chans=input_channel)
        self.convnext3 = convnext.ConvNeXt(in_chans=input_channel)
    
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

        return result
        
