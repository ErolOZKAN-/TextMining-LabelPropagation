import networkx as nx
import heapq,itertools
import csv

def parse(filename):
    reader = csv.reader(open(filename, 'r'), delimiter=',')
    data = [row for row in reader]
    print "Reading and parsing the data into memory..."
    return data

def parseDirectedGraph(data,sentencesList,kNeightestValue):
    DG = nx.Graph()

    for i in range(0,len(data)):
        sentence = sentencesList[i][sentencesList[i].find('\t')+1:]
        label = sentencesList[i][:sentencesList[i].find('\t')]
        DG.add_node('N'+str(i), sentence=sentence,label = label)

    for i, row in enumerate(data):
        lst = [float(x) for x in row]
        nlargest = heapq.nlargest(kNeightestValue, zip(lst, itertools.count()))
        for column,j in nlargest:
            column = float (column)
            if(column>0.0):
                DG.add_edge('N'+str(i),'N'+str(j),weight = column)

    return DG





