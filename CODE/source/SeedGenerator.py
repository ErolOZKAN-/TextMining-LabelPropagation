from collections import defaultdict,Counter
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from networkx import to_scipy_sparse_matrix
from myGraph import *
from SeedGenerator import *
import time
from datetime import datetime
from graphPrinter import printGraph, getLabelsAndColour,getLabelsAndColourFromGraph
from itertools import chain

def getR8TrainSentences(trainFileName):
    with open(trainFileName, "r") as ins:
        sentencesList = []
        for line in ins:
            sentencesList.append(line)
    return sentencesList

def generateSeed(sentencesList,modNumber):
    seedList = list()
    for i,sentence in enumerate(sentencesList):
        if(i%modNumber == 1):
            seedList.append(('N'+str(i),sentencesList[i],1))
    return seedList

def printToFile(seedList,fileName):
    f = open(fileName, 'w')
    for i,sentence in enumerate(seedList):
        f.write(sentence[0]+'\t'+(sentence[1][:sentence[1].find('\t')])+'\t'+'1.0'+'\n')
    f.close()

def printToSomeOtherFile(goldenLabels,fileName):
    f = open(fileName, 'w')
    labelCounts = list()
    for dict in goldenLabels:
        labelCounts.append(goldenLabels[dict])
    info = Counter(labelCounts)
    for i in info:
        f.write(str(i)+"\t"+str(info[i])+"\t%"+str(float(info[i])*100/len(goldenLabels)))
        f.write("\n")
    f.close()