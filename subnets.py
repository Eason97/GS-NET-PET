import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layer = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU())
    return layer

def conv_3d_ista(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layer = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    )
    return layer

def soft_thresholding(x, threshold):
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)

class Sino_modify_net(nn.Module):
    def __init__(self):
        super(Sino_modify_net, self).__init__()
        self.layer1 = conv_3d(1, 32)
        self.layer2 = conv_3d(32, 32)
        self.layer3 = conv_3d(32, 1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_out = x + x3
        return x_out


class F_transform(nn.Module):
    def __init__(self):
        super(F_transform, self).__init__()
        self.layer1 = conv_3d_ista(32,32)

    def forward(self, x):
        x1 = self.layer1(x)
        return x1

class F_inverse(nn.Module):
    def __init__(self):
        super(F_inverse, self).__init__()
        self.layer1 = conv_3d_ista(32,32)

    def forward(self, x):
        # Implement the inverse operation accordingly
        x1 = self.layer1(x)
        return x1
