import json
import os
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def make_image_pairs(dirname):
    posts = []
    if os.path.isdir(dirname):
        with open(dirname + '/'+ dirname+'.json') as data_file:
            data = json.load(data_file)

        for x in range(len(data)):
            image= []
            #image.append((data[x]["thumbnail_src"]).rsplit('/', 1)[-1])
            if (data[x]['edge_media_to_caption']['edges']):
                image.append((data[x]["thumbnail_src"]).rsplit('/', 1)[-1])
                image.append(data[x]['edge_media_to_caption']['edges'][0]['node']['text'])
                posts.append(image)
    return posts


def import_images(textFile, mod = False):
    image = []
    users = open(textFile)

    if ( not mod ):
        for user in users:
            if os.path.isdir(user[:-1]):
                with open(user[:-1] + '/'+ user[:-1]+'.json') as data_file:
                    data = json.load(data_file)

                for x in range(1, len(data)):
                    #image.append((data[x]["thumbnail_src"]).rsplit('/', 1)[-1])
                    if (data[x]['edge_media_to_caption']['edges']):
                        image.append(user[:-1] +'/' + (data[x]["thumbnail_src"]).rsplit('/', 1)[-1])
        image = [ x for x in image if(os.path.isfile(x))]
        image = [ mpimg.imread(dirName) for dirName in image[1:] ]
    else:
        image = [ mpimg.imread("../resized/"+ filename) for filename in os.listdir("../resized/")]
    return image


X = import_images('users.txt', mod = True)
print(X)
plt.imshow(X[1])
plt.show()


def save_mod_images(textFile):
    image = []
    users = open(textFile)

    for user in users:
        if os.path.isdir(user[:-1]):
            with open(user[:-1] + '/'+ user[:-1]+'.json') as data_file:
                data = json.load(data_file)

            for x in range(1, len(data)):
                if (data[x]['edge_media_to_caption']['edges']):
                    image.append(user[:-1] +'/' + (data[x]["thumbnail_src"]).rsplit('/', 1)[-1])
    image = [ x for x in image if(os.path.isfile(x))]
    for dirImg in image:
        img = Image.open(dirImg)
        img.thumbnail((600,600), Image.ANTIALIAS)
        img.save("../resized/"+(dirImg).rsplit('/',1)[-1], "JPEG")
        img.close()
    return image
#save_mod_images('users.txt')



def make_nonVec_data_set(textFile):
    user_list = open(textFile)
    w =[make_image_pairs(x[:-1]) for x in user_list]
    
    sent = []
    for user in w:
        for comment in user:
            sent.append(comment)
    user_list.close()
    return sent

image_pairs =make_nonVec_data_set('users.txt')


def getSentences(nonVec_dataset):
    sentences = [ [i[1]] for i in nonVec_dataset]
    return sentences

def getVocab(nonVec_dataset):
    raw = ""

    for i in nonVec_dataset:
        raw +=" " +  i[1]
    


    raw = raw.replace(".", "")
    raw = raw.replace("!", "")
    raw = raw.replace("?", "")
    raw = raw.replace(",", "")
    raw = raw.replace("(", "")
    raw = raw.replace(")", "")


    vocab = set(raw.split())
    return vocab

vocab=getVocab(image_pairs)


def createModel(sentences):
    worded = [ i[0].split() for i in sentences ]
    model = Word2Vec(worded, min_count=1)
    #model.save('model.bin')
    #model.save('../model.bin')

    return model


sentences = getSentences(image_pairs)
model = createModel(sentences)




#for i in pr:
#    print(i)
#print(len(pr))
#print("Manchester" in pr )
#print( "hope" in pr)
