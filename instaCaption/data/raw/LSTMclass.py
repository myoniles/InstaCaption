import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

####################################################################################################
#   LSTM Caption Class
#   Takes the output dimensions of the CNN, the embedding dim (to resize), and hidden dimensions
#   
#   Because the CNN output is so big its reduced using a single linear layer to embed size
#   the previous word is concat. to CNN out producing a tensor of embed*2
#   finally the output tensor is connected to the WordVec space thru 1 linear layer
####################################################################################################


class LSTMcaption(nn.Module):
    def __init__(self, CCNoutput_dim,  embedding_dim, hidden_dim):
        super(LSTMcaption, self).__init__()
        self.hidden_dim = hidden_dim
        self.CNNdim = CCNoutput_dim
        self.embed = embedding_dim

        self.lstm = nn.LSTM( self.embed*2, self.hidden_dim, 3 ).cuda()
        self.linear = nn.Linear(self.hidden_dim, embedding_dim).cuda()
        self.prev = Variable(torch.zeros(100)).cuda()
        self.i2v = nn.Linear(self.CNNdim , embedding_dim).cuda()
        self.lstm.cuda()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return(Variable(torch.zeros(3,1,100)).cuda(), Variable(torch.zeros((3,1,100))).cuda())
    def forward(self, imageVec, word):
        imageVec = self.i2v(imageVec)
        imageVec=torch.cat((imageVec.view(100),self.prev), 0).unsqueeze(1)
        
        self.prev = Variable(word).cuda()
        outword, self.hidden = self.lstm(imageVec, self.hidden)
        outword = self.linear(outword)
        

        return outword
    #This output is put through tanh on the other side
