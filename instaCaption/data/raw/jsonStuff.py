import json
import os
import re

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
    '''
    for line in user_list:
        w = w + make_image_pairs(line[:-1])
    '''
    return w

heck =make_nonVec_data_set('users.txt')
#print(heck)

def getVocab(nonVec_dataset):
    raw = ""

    for j in nonVec_dataset:
        for i in j:
            raw +=" " +  i[1]
    print(raw)

    raw = raw.replace(".", "")
    raw = raw.replace("!", "")
    raw = raw.replace("?", "")
    raw = raw.replace(",", "")
    raw = raw.replace("(", "")
    raw = raw.replace(")", "")


    vocab = set(raw.split())
    return vocab

pr=getVocab(heck)

for i in pr:
    print(i)
print(len(pr))
#print("Manchester" in pr )
#print( "hope" in pr)
