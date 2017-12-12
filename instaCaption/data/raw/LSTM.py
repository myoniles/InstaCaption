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
#   Look at the LSTMclass.py file for more information
####################################################################################################

print("[ ]  Creating LSTM...\r", end="")
model = LSTMcaption(1000,100, 100)

print("[✔️]")



####################################################################################################
#   Define loss function:
#       L1 loss: |x-t|
#       tanh: (e^(2x) -1)/(e^(2x) +1)
#       Adam optimizer
#   totalLoss: keeps track of the total losss accumulated to find average
#   iteration: how many times the net has guesses a word also for average
#   epochs: take a wild guess buddy
####################################################################################################

loss_fun = nn.L1Loss() 
optimizer = optim.Adam(model.parameters(), lr =0.0001)
tanh = nn.Hardtanh()

totalLoss = 0
iteration = 1
epochs = 20

####################################################################################################
#   For the number of epochs:
#       for each data point in the data set:
#           create the output vector from the CNN
#           
#           set the gradient of all model parameters to zero
#           zero the initial state of the LSTM
#       
#           for word in the sentence of the datapoint:
#               heck is for counting which word we are at in the sentence
#               
#               target is meant to hold the vector representation of the word at position heck
#               this is stored in target
#               
#               out is the output of the LSTM
#                   this takes the CNN output and the current word as an input
#               GUESS WHAT LOSS REPRESENTS
#                   activate output of CNN with tanh
#                   indexing to make both [100] tensors
#
#               The next few lines are for making the Print statement look nice
#                   Loss: loss      avgLoss: Average Loss   image number: i 
#                   word: guess     c: confidence in W2V    T: Target word
#               
#               Backprop loss
#               optimizer step
#       save per epoch
####################################################################################################
for j in range(epochs):
    for i in range(len(jsonStuff.Ximproved)):
        input_var = Variable(image_data[i].unsqueeze(0))
        CNN_out = CNNmodel(input_var)
        CNN_out = CNN_out.unsqueeze(0)

        model.zero_grad()
        model.hidden = model.init_hidden() 
    
        for word in Y[i]:
            heck = 0

            temp =  torch.from_numpy(word)
            target=Variable(temp).cuda()

            out = model(CNN_out, temp)
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
    torch.save(model, '../LSTM2.pt')



