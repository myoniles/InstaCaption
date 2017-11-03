
def name_2_vec(name):
    nameVec =[0] * 27
    name = name.lower()
    name = [ord(l)-97 for l in name] 

    for c in name:
        if(c > 0):
            nameVec[c] += 1
        else:
            nameVec[26] += 1
    return nameVec

#print(name_2_vec("Michael"))


def char_2_vec(name):
    X = [] 
    name = name.lower()
    name = [ord(l) - 97 for l in name]

    for c in name:
        charVec = [0] * 27
        if( c > 0 ):
            charVec[c] += 1
        else:
            charVec[26] +=1
        X.append(charVec)
    return X

#print(char_2_vec("Michael"))

def list_2_vec(fileF):
    open(fileF,'r')
    x = []
    for line in fileF:
        name = char_2_vec(line)
        for c in name:
            x.append(c)
    return x

#print(list_2_vec('testDatapy.txt'))

def lowerfile(fil):
    g = open(fil, 'r+')
    names = []
    for line in g:
        names.append(line.lower())
    g.close()

    g = open(fil, 'w')
    for n in names:
        g.write(n)
    g.close()

#lowerfile('male.txt')
#lowerfile('female.txt')

def combinefiles(file1, file2):
    files = [file1, file2]
    names = []
    for f in files:
        g = open(f , 'r')
        for line in g:
            names.append(line.lower())
        g.close()

    names.sort()

    g = open('combined.txt' , 'w')
    for n in names:
        g.write(n)
    g.close()

#combinefiles("male.txt", "female.txt")


def makeYvec():
    m = open('male.txt', 'r').readlines()
    c = open('combined.txt', 'r').readlines()
    w = []
    
    for i in c:
        if any(i in s for s in m):
            w.append([1])
        else:
            w.append([0])
    return (w)


print(makeYvec())

