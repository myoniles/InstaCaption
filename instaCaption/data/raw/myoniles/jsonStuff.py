import json
import os
'''
with open('myoniles.json') as data_file:
    data = json.load(data_file)


print(len(data[0]['edge_media_to_caption']['edges']))
print(data[0]['edge_media_to_caption']['edges'][0]['node']['text'])

posts= []


for x in range(len(data)):
    image= []
    image.append((data[x]["thumbnail_src"]).rsplit('/', 1)[-1])
    if (data[x]['edge_media_to_caption']['edges']):
        image.append(data[x]['edge_media_to_caption']['edges'][0]['node']['text'])
    posts.append(image)

#print(posts)
'''

def make_image_pairs(dirname):
    if not os.path.isdir(dirname):
        print("Ya doofus")
        quit()
    posts = []
    
    with open((dirname+'.json')) as data_file:
        data = json.load(data_file)

    for x in range(len(data)):
        pair = []
        image.append((data[x]["thumbnail_src"]).rsplit('/', 1)[-1])
        if (data[x]['edge_media_to_caption']['edges']):
            image.append(data[x]['edge_media_to_caption']['edges'][0]['node']['text'])
        posts.append(pair)
    return posts

print(make_image_pairs('.'))
