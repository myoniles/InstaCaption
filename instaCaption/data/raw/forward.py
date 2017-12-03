import torch
import torch.nn as nn
import torch.models as models
import torchvision.transforms as tranforms
import jsonStuff
from gensim.models import Word2Vec
import matplotlib.pylot as plt
import matplotlib.image as mpimg


####################################################################################################
#                              Insert the directory of the image below                             #
####################################################################################################
img_location = ""

try:
    heck = mpimg.imread(img_location)
except:
    print("Not a valid image location")
    quit()

heck = transforms.ToTensor(heck)


