import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import jsonStuff
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.autograd import Variable
import pickle
from LSTMclass import LSTMcaption
import torch.backends.cudnn as cudnn


####################################################################################################
#                              Insert the directory of the image below                             #
####################################################################################################
img_location = "../resized/22802224_160937671169750_7214613822470881280_n.jpg"

cudnn.benchmark = True


try:
    heck = mpimg.imread(img_location)
except:
    print("Not a valid image location")
    quit()

tens = transforms.ToTensor()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tanh = nn.Hardtanh()


heck = tens(heck)
heck = norm(heck)

CNNmodel = models.resnet50(pretrained=True).cuda()
CNNmodel = torch.nn.DataParallel(CNNmodel).cuda()
heck = Variable(heck).unsqueeze(0)


heck.cuda()
lstmIn = CNNmodel(heck)

lstm = torch.load("LSTM.pt")

sent = ""

while ( "endSent" not in sent ):
    word = lstm(lstmIn.unsqueeze(0))

    #print(tanh(word))
    word = word.cpu()
    word = tanh(word)
    word = word.data.numpy()
    word = word[0][0]

    toWord = jsonStuff.model.wv.most_similar(positive=[word])
    print(toWord[0], "\r", end="")
    sent+= toWord[0][0] + " "

print(sent)
