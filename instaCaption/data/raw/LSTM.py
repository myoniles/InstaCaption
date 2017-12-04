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
import torch.optim as optim
import jsonStuff
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from LSTMclass import LSTMcaption
from gensim.models import Word2Vec

print("[✔️]")

cudnn.benchmark = True


#Define pretrained Resnet Model
print("[ ]  Importing RESNET...\r", end="")
CNNmodel = models.resnet50(pretrained=True).cuda()
CNNmodel = torch.nn.DataParallel(CNNmodel).cuda()
print("[✔️]")


####################################################################################################
#   The CNN Model takes an image ( normalized below ) and returns 
#   a 1 x 1000 float tensor
####################################################################################################

print("[ ]  normalizing data...\r", end ="")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_data = [i[0] for i in jsonStuff.Ximproved]
Y = [i[1] for i in jsonStuff.Ximproved]
####################################################################################################
#   first normalize the images to the CNN's specifications
#   then makes a list of image numpy vectors.
#   these arrays are 3 x L x W where neither L nor W are larger than 300
####################################################################################################

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
image_data = [normalize(i) for i in image_data]

#plt.imshow(toPIL(image_data[557]))
#plt.show() # """"""""""""" NORMALIZED """""""""""""""



#plt.imshow(toPIL(image_data[557]))
#plt.show() # """"""""""""" NORMALIZED """""""""""""""

print("[✔️]  hihowareya?        ")

####################################################################################################
#   Custom LSTMCaption Class:
#       takes CCN output dimensions, the output dimension, and the mapping dimension
#       hidden returns a clear initialized hidden layer
#       redefines foward pass
#       moves things to GPU
####################################################################################################


#class LSTMcaption(nn.Module):
#    def __init__(self, CCNoutput_dim,  embedding_dim, hidden_dim):
#        super(LSTMcaption, self).__init__()
#        self.hidden_dim = hidden_dim
#        self.CNNdim = CCNoutput_dim
#        self.lstm = nn.LSTM(self.CNNdim, self.hidden_dim).cuda()
#
#        self.lstm.cuda()
#        self.hidden = self.init_hidden()
#        #self.long_2_word = nn.Linear(hidden_dim, embedding_dim)
#        
#        
#    def init_hidden(self):
#        return(Variable(torch.rand(1,1,100)).cuda(), Variable(torch.randn((1,1,100))).cuda())
#    def forward(self, imageVec):
#        outLong, self.hidden = self.lstm(imageVec, self.hidden)
#        #Outword = self.long_2_word(outLong, -1)
#        return outLong

print("[ ]  Creating LSTM...\r", end="")
model = LSTMcaption(1000,100, 100)

print("[✔️]")




loss_fun = nn.L1Loss() 
optimizer = optim.SGD(model.parameters(), lr =0.0001)
tanh = nn.Hardtanh()


inputVar = Variable(image_data[0])
#print('CNN', CNNmodel(inputVar))



totalLoss = 0
iteration = 1
epochs = 20

for j in range(epochs):
    for i in range(len(jsonStuff.Ximproved)):
        input_var = Variable(image_data[i].unsqueeze(0))
    
        #plt.imshow(toPIL(image_data[i]))
        #plt.show()
    

        CNN_out = CNNmodel(input_var)
        CNN_out = CNN_out.unsqueeze(0)

        model.zero_grad()

        model.hidden = model.init_hidden() 
    
        for word in Y[i]:
            heck = 0
            temp= torch.from_numpy(word)
            temp = temp.cuda()
        
            
            #print(lstm.hidden) 

            target=Variable(temp)
            out = model(CNN_out)
        
        

            loss = loss_fun(tanh(out[0][0]), target)
            lossp = loss.data[0]
            totalLoss += lossp
            wordp = jsonStuff.model.wv.most_similar(positive=[tanh(out[0][0]).data.cpu().numpy()])
            wordt = jsonStuff.model.wv.most_similar(positive=[word])
            print("loss %.2f \tavgLoss %.2f \timage number: %d \tword: (%.5s) \tc: %.3f \tT: %.5s\r"  % (lossp, (totalLoss/iteration), i, wordp[0][0], wordp[0][1], wordt[0][0] ), end = "")  

            if( heck == 0):
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            heck += 1
            iteration +=1
            optimizer.step()
    torch.save(lstm, '../LSTM2.pt')



