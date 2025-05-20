import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layer = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU())
    # init.xavier_uniform_(layer[0].weight)
    # if layer[0].bias is not None:
    #     nn.init.constant_(layer[0].bias, 0)
    return layer

def conv_3d_ista(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layer = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    )
    # init.xavier_uniform_(layer[0].weight)
    # if layer[0].bias is not None:
    #     nn.init.constant_(layer[0].bias, 0)
    # init.xavier_uniform_(layer[2].weight)
    # if layer[2].bias is not None:
    #     nn.init.constant_(layer[2].bias, 0)
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
        # x1=x+x1
        return x1

class F_inverse(nn.Module):
    def __init__(self):
        super(F_inverse, self).__init__()
        self.layer1 = conv_3d_ista(32,32)


    def forward(self, x):
        # Implement the inverse operation accordingly
        x1 = self.layer1(x)
        # x1 = x + x1
        return x1

if __name__ == '__main__':
    data=torch.randn(2,2,5,96,96).cuda()
    model=F_inverse().cuda()
    output=model(data)
    print(output.shape)