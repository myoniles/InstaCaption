import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

#I used the pre installed imdb data set in tensorflow
train, test, _ = imdb.load_data(path ='imdb.pkl', n_words=22500, valid_portion=0.1)

trainX, trainY = train
testX, testY = test

trainX = pad_sequences(trainX, maxlen=150, value=0.)
testX = pad_sequences(testX, maxlen=150, value=0.)

trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

net = tflearn.input_data([None, 150])
net = tflearn.embedding(net, input_dim= 22500, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.5)
net = tflearn.fully_connected(net, 2)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, learning_rate=0.0001)

#train
model = tflearn.DNN(net)
model.fit(trainX, trainY, validation_set =(testX, testY), n_epoch=15,  show_metric=True, batch_size=25)
 

model.save("avRNN.tfl")

