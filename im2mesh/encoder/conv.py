import torch
import torch.nn as nn

# import torch.nn.functional as F
from torchvision import models
from im2mesh.common import normalize_imagenet
from .convnext.convnextv2 import convnextv2_atto, convnextv2_tiny


class ConvEncoder(nn.Module):
    r"""Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    """

    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 512, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


class Resnet18(nn.Module):
    r"""ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError("c_dim must be 512 if use_linear is False")

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet34(nn.Module):
    r"""ResNet-34 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError("c_dim must be 512 if use_linear is False")

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet50(nn.Module):
    r"""ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError("c_dim must be 2048 if use_linear is False")

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet101(nn.Module):
    r"""ResNet-101 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet101(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError("c_dim must be 2048 if use_linear is False")

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class ConvNeXtTiny(nn.Module):
    r"""ConvNeXtTiny encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear

        self.features = models.convnext_tiny(pretrained=True)
        self.features.classifier = nn.Sequential()

        self.fc = nn.Linear(768, c_dim)

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net.reshape(-1, 768))
        return out


class ConvNeXt2Tiny(nn.Module):
    r"""ConvNeXt2Tiny encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear

        self.features = convnextv2_tiny()
        self.features.load_state_dict(
            torch.load("convnextv2_tiny_1k_224_ema.pt"), strict=False
        )
        self.features.head = nn.Sequential()
        self.fc = nn.Linear(768, c_dim)

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net.reshape(-1, 768))
        return out


class ConvNeXtTinyFP(nn.Module):
    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize

        self.conv1 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)

        self.path0 = nn.Conv2d(3, 3, kernel_size=1)
        self.path1 = nn.Conv2d(3, 3, kernel_size=1)

        self.enc0 = ConvNeXtTiny(c_dim, normalize=False)
        self.enc1 = ConvNeXtTiny(c_dim, normalize=False)
        self.enc2 = ConvNeXtTiny(c_dim, normalize=False)

        self.pool = nn.AvgPool1d(3)

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)

        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        x1 = self.up2(x2) + self.path1(x1)
        x0 = self.up1(x1) + self.path0(x0)

        x2 = self.enc2(x2)
        x1 = self.enc1(x1)
        x0 = self.enc0(x0)

        x = torch.concat([x0.unsqueeze(2), x1.unsqueeze(2), x2.unsqueeze(2)], dim=2)
        x = self.pool(x).squeeze(2)

        return x


class ConvNeXtTinyFPMax(nn.Module):
    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize

        self.conv1 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=4, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)

        self.path0 = nn.Conv2d(3, 3, kernel_size=1)
        self.path1 = nn.Conv2d(3, 3, kernel_size=1)

        self.enc0 = ConvNeXtTiny(c_dim, normalize=False)
        self.enc1 = ConvNeXtTiny(c_dim, normalize=False)
        self.enc2 = ConvNeXtTiny(c_dim, normalize=False)

        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)

        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        x1 = self.up2(x2) + self.path1(x1)
        x0 = self.up1(x1) + self.path0(x0)

        x2 = self.enc2(x2)
        x1 = self.enc1(x1)
        x0 = self.enc0(x0)

        x = torch.concat([x0.unsqueeze(2), x1.unsqueeze(2), x2.unsqueeze(2)], dim=2)
        x = self.pool(x).squeeze(2)

        return x
