print("Importing... ALL THE PACKAGES", end = " ")
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torch.optim as optim
import jsonStuff
import matplotlib.image as mpimg
import matplotlib.pyplot as plt



print("done")

cudnn.benchmark = True


#Define pretrained Resnet Model
print("importing RESNET...", end=" ")
CNNmodel = models.resnet152(pretrained=True)
CNNmodel = torch.nn.DataParallel(CNNmodel).cuda()
print("done")


print("normalizing data...", end =" ")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_data = datasets.ImageFolder("../resized",transforms.Compose([transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
print("hihowareya?")


img_loader=torch.utils.data.DataLoader(image_data)

class LSTMcaption(nn.Module):
    def __init__(self,CCNoutput_dim,  embedding_dim, hidden_dim, W2VModel):
        super(LSTMcaption, self).__init__()
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embedding_dim)

        self.lstm(CNNoutput_dim, embedding_dim)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return((Variable(torch.rand(1,1,100)), Variable(torch.randn((1,1,100)))))
    def forward(self, imageVec):
        outWord, self.hidden = self.lstm(imageVec, self.hidden)
        return outWord

print("Creating LSTM...", end=" ")
lstm = nn.LSTM(1000,100)
lstm.cuda()
print("done")
hidden = (Variable(torch.rand(1,1,100)), Variable(torch.randn((1,1,100))))

loss_fun = nn.NLLLoss()
optimizer = optim.SGD(lstm.parameters(), lr =0.1)






for i, (input, target) in enumerate(img_loader):
    input_var = Variable(input)

    print(input_var)

    


    imgplot = plt.imshow(input_var.data.numpy())
    plt.show()

    break

    '''
    CNN_out = CNNmodel(input_var)

    

    lstm.zero_grad()

    lstm.hidden = model.init_hidden()

    target 
'''

