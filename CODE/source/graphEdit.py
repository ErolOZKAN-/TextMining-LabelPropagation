import numpy as np

similarityFilenameToParse = '../data/assgn4Data/erol2.mat'
input = np.loadtxt(similarityFilenameToParse, dtype='i', delimiter=',')

newMatrix = np.zeros(shape=(len(input),len(input)))

for i, row in enumerate(input):
    for j,value in enumerate(row):
        newMatrix[i][j]=float(input[i][j])/10

print newMatrix

np.savetxt('../data/assgn4Data/erol.mat',newMatrix, delimiter=',')