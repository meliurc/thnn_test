# a jupyter notebook test pytorch CNN output

# ----------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ----------------------------------------------------------------

# 2 input image channel, 2 output channels, 2x2 square convolution kernel
kernelTensor = torch.ones([2, 2, 2, 2], out=None)
conv1 = nn.Conv2d(2, 2, 2)
conv1.weight.data = kernelTensor

inputArr = np.array(range(0, 18)).reshape(1, 2, 3, 3)
inputTensor = torch.from_numpy(inputArr).float()

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 2 input image channel, 2 output channels, 2x2 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 2, 2)

    def forward(self, x):
        x = self.conv1(x)

        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data = m.weight.data.new_ones(m.weight.data.size()).float()
        print m.weight.data

# ----------------------------------------------------------------

net = Net()
net.apply(weights_init)

out = net(inputTensor)
print(out)