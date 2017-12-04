import random

random.seed(22)

numbers = []
Yvect = []

def toBinary(decNum):
    binNum = []
    for exp in range(7,-1,-1):
        if decNum >= 2**exp:
            binNum.append(1)
            decNum = decNum - (2**exp)
        else:
            binNum.append(0)
    return binNum



def makeVectors():
    numbers = []
    Yvect = []
    for i in range(256):
        b= toBinary(i)
        numbers.append(b)
        if (i % 3 == 0):
           Yvect.append([1])
        else:
            Yvect.append([0])
    return numbers, Yvect
