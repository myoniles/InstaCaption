import numpy as np
import torch
from torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNN(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 18, 15)
        self.conv2 = nn.Conv2d(18,
