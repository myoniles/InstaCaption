print("[ ]  Importing... ALL THE PACKAGES\r", end = " ")
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
print("[ ]  Importing RESNET...\r", end=" ")
CNNmodel = models.resnet152(pretrained=True)
CNNmodel = torch.nn.DataParallel(CNNmodel).cuda()
print("[✔️]")








print("[ ] normalizing data...\r", end =" ")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_data = [i[0] for i in jsonStuff.Ximproved]
Y = [i[1] for i in jsonStuff.Ximproved]
        

#plt.imshow(image_data[0])
#plt.show()
toTens = transforms.ToTensor()
toPIL = transforms.ToPILImage()




image_data = [ toTens(i) for i in image_data ]
normalize(image_data)
image_data = [i.unsqueeze(0) for i in image_data]



#plt.imshow(toPIL(image_data[557]))
#plt.show() # """"""""""""" NORMALIZED """""""""""""""

print("[✔️] hihowareya?")





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

print("[ ]  Creating LSTM...\r", end=" ")
lstm = nn.LSTM(1000,100)
lstm.cuda()
print("[✔️]")
hidden = (Variable(torch.rand(1,1,100)), Variable(torch.randn((1,1,100))))

loss_fun = nn.L1Loss()
optimizer = optim.SGD(lstm.parameters(), lr =0.1)



inputVar = Variable(image_data[0])
print(CNNmodel(inputVar))




for i in range(len(jsonStuff.Ximproved)):
    input_var = Variable(image_data[i])
    CNN_out = CNNmodel(input_var)
    CNN_out = CNN_out.unsqueeze(0)
    lstm.zero_grad()

    lstm.hidden = (Variable(torch.rand(1,1,100)), Variable(torch.randn((1,1,1000)))) 

    print(CNN_out)
    for word in Y[i]:
        temp= torch.from_numpy(word)
        target=Variable(temp, requires_grad=False)
        out = lstm(CNN_out, lstm.hidden)

        
        #print(out[0])
        out = out[0]
        
        

        target.cpu() 

        loss = loss_fun(out, target.cpu())
        loss.backward()
        optimizer.step()



