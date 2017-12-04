import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn



class LSTMcaption(nn.Module):
    def __init__(self, CCNoutput_dim,  embedding_dim, hidden_dim):
        super(LSTMcaption, self).__init__()
        self.hidden_dim = hidden_dim
        self.CNNdim = CCNoutput_dim
        self.lstm = nn.LSTM(self.CNNdim, self.hidden_dim).cuda()
        self.linear = nn.Linear(self.hidden_dim, embedding_dim).cuda()


        self.lstm.cuda()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        self.hidden =(Variable(torch.zeros(1,1,100)).cuda(), Variable(torch.zeros((1,1,100))).cuda())
        return(Variable(torch.rand(1,1,100)).cuda(), Variable(torch.randn((1,1,100))).cuda())
    def forward(self, imageVec):
        outlstm, self.hidden = self.lstm(imageVec, self.hidden)

        outword = self.linear(outlstm)
        

        return outword

