import torch
from torch import np

def sigmoid(x):
    return 1/(1np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

np.random.seed(1)

W_i2h = 2* np.random.random((8,4)) -1
W_h2o = 2* np.random.random((4,1)) -1




