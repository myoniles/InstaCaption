from data import toBinary, makeVectors
import numpy as np

#The sigmoid and the derivate of the sigmoid function are used for regularization respectively
def sigmoid(value):
    return np.exp(value)/(np.exp(value)+1)

def dsigmoid(value):
    return value * (1 - value)

# This fuction is from the data file and produces an array of 8-bit representations of numbers
X, Y = makeVectors()
X = np.array(X)
Y = np.array(Y)


### NOTICE THAT BIAS HAS NOT BEEN IMPLEMENTED
### initializes random weights for the net to produce a single probability
#bias = 1
W_i2h = 2* np.random.random((8,4)) -1
W_h2o = 2* np.random.random((4,1)) -1
#B_i2h = 2* np.random.random((8,4)) -1
#B_h2o = 2* np.random.random((4,1)) -1
learning_rate = 0.01


for i in range ( 250000 ):
    # This is what trains the nueral net and updates the weights
    inputV = X
    # forward pass
    unsq_h = np.dot(inputV, W_i2h) 
    sqh = sigmoid(unsq_h)

    unsq_o = sqh.dot(W_h2o) 
    sqo = sigmoid(unsq_o)

    #find loss
    loss = Y - sqo

    #for documentation purposes
    if ( i % 2000 == 0 ):
        print ("Error:", str(np.mean(np.abs(loss))), "\r", end ="")

    #find deltas
    dW_h2o = loss * dsigmoid(sqo)

    dW_i2h = (dW_h2o.dot(W_h2o.T)) * dsigmoid(sqh)

    #Update the weights
    W_i2h += inputV.T.dot(dW_i2h) * learning_rate
    W_h2o += sqh.T.dot(dW_h2o) * learning_rate

def predict(int_G):
    #This just does a forward pass and returns the squashed value of the output layer
    #its a repeat of the first half of the loop
    x = toBinary(int_G)
    unsq_h = np.dot(x, W_i2h) 
    sqh = sigmoid(unsq_h)

    unsq_o = sqh.dot(W_h2o)
    sqo = sigmoid(unsq_o)
    return sqo[0]


testing = True
while(testing):
    testInput = input("Enter a number")
    testInput = int(testInput)
    if(testInput >= 0 and testInput < 256 ):
        if(predict(testInput) >= 0.5):
            heck = "yes"
        else:
            heck = "no"
        print("The probability that" , testInput , "is divisible by three is" , predict(testInput))
        print("In other words:" , heck)
    else:
        testing = False

