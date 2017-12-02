print("[ ]  Importing... ALL THE PACKAGES\r", end = "")
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
print("[✔️]")

cudnn.benchmark = True


#Define pretrained Resnet Model
print("[ ]  Importing RESNET...\r", end="")
CNNmodel = models.resnet152(pretrained=True).cuda()
CNNmodel = torch.nn.DataParallel(CNNmodel).cuda()
print("[✔️]")


print(CNNmodel)
####################################################################################################
#   The CNN Model takes an image ( normalized below ) and returns 
#   a 1 x 1000 float tensor
####################################################################################################

print("[ ] normalizing data...\r", end ="")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_data = [i[0] for i in jsonStuff.Ximproved]
Y = [i[1] for i in jsonStuff.Ximproved]
####################################################################################################
#   first normalize the images to the CNN's specifications
#   then makes a list of image numpy vectors.
#   these arrays are 3 x L x W where neither L nor W are larger than 300
####################################################################################################

#plt.imshow(image_data[0])
#plt.show()



toTens = transforms.ToTensor()
toPIL = transforms.ToPILImage()

####################################################################################################
#   To Tens is used to turn the Numpy arrays into pytorch tensors
#   to PIL is used when viewing images
#   
#   For some reason the tensors created by ToTens are two dimensional
#   unsqueeze is used to add an additional layer
####################################################################################################
image_data = [ toTens(i) for i in image_data ]
normalize(image_data)
image_data = [i.unsqueeze(0) for i in image_data]



#plt.imshow(toPIL(image_data[557]))
#plt.show() # """"""""""""" NORMALIZED """""""""""""""

print("[✔️]  hihowareya?      ")

####################################################################################################
#   Custom LSTMCaption Class:
#       takes CCN output dimensions, the output dimension, and the mapping dimension
#       hidden returns a clear initialized hidden layer
#       redefines foward pass
#       moves things to GPU
####################################################################################################


class LSTMcaption(nn.Module):
    def __init__(self, CCNoutput_dim,  embedding_dim, hidden_dim):
        super(LSTMcaption, self).__init__()
        self.hidden_dim = hidden_dim
        self.CNNdim = CCNoutput_dim
        self.lstm = nn.LSTM(self.CNNdim, self.hidden_dim).cuda()

        self.lstm.cuda()
        self.hidden = self.init_hidden()
        #self.long_2_word = nn.Linear(hidden_dim, embedding_dim)
        
        
    def init_hidden(self):
        return(Variable(torch.rand(1,1,100)).cuda(), Variable(torch.randn((1,1,100))).cuda())
    def forward(self, imageVec):
        outLong, self.hidden = self.lstm(imageVec, self.hidden)
        #Outword = self.long_2_word(outLong, -1)
        return outLong

print("[ ]  Creating LSTM...\r", end="")
lstm = LSTMcaption(1000,100, 100)

print("[✔️]")




loss_fun = nn.NLLLoss()
optimizer = optim.SGD(lstm.parameters(), lr =0.1)



inputVar = Variable(image_data[0])
#print('CNN', CNNmodel(inputVar))




for i in range(len(jsonStuff.Ximproved)):
    input_var = Variable(image_data[i])
    CNN_out = CNNmodel(input_var)
    print(CNN_out)
    CNN_out = CNN_out.unsqueeze(0)

    lstm.zero_grad()

    lstm.hidden = lstm.init_hidden() 

    for word in Y[i]:
        temp= torch.from_numpy(word)
        print('short temp', temp)
        temp = temp.type(torch.LongTensor)
        temp = temp.cuda()

        print('temp', temp)

        target=Variable(temp, requires_grad=False)
        out = lstm(CNN_out)
        target.type(torch.LongTensor)
        
        print(out[0])
        
        
        
        

        

        loss = loss_fun(out, target[0])
        if( i == 0):
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        optimizer.step()
