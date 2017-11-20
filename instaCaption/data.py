# HECK ALL OF THIS YOU BIG DUMMY IMPORT AND VECTORIZE ON THE FLY
# HECK IT WE'LL DO IT LIVE

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

'''
def load_ims (filename):
    X = []

    if( not os.path.isdir(dirname)):
        quit()

    for filename in os.listdir(dirname):
        paths = dirname+ "/" +filename
        img = mping.imread(paths)
        X.append(img)
    return X
'''

def load_im(filename,w=640, h=640):
    filename = "data/"+filename
    img = Image.open(filename)
    img.thumbnail((h, w),Image.ANTIALIAS)
    img.save(filename, "JPEG")
    img = mpimg.imread(filename)
    return img

plt.imshow(load_im('nom.jpg'))
plt.show()
