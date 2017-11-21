import json
import os
from gensim.models import Word2Vec

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

def make_nonVec_data_set(textFile):
    user_list = open(textFile)
    w =[make_image_pairs(x[:-1]) for x in user_list]
    
    sent = []
    for user in w:
        for comment in user:
            sent.append(comment)
    
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
    model.save('model.bin')
    model.save('../model.bin')

    return model


sentences = getSentences(image_pairs)
model = createModel(sentences)


#for i in pr:
#    print(i)
#print(len(pr))
#print("Manchester" in pr )
#print( "hope" in pr)
