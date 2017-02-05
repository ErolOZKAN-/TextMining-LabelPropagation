import networkx as nx
import matplotlib.pyplot as plt

colourValues = {"acq": "b",
               "crude": "g",
               "earn": "r",
               "grain": "c",
               "interest": "m",
               "money-fx": "y",
               "ship": "k",
               "trade": "k",
               " -": "w"}

def getLabelsAndColourFromGraph(myGraph):
    colorValues = list();
    nodeLabels={}
    for n,d in myGraph.nodes_iter(data=True):
        colour = colourValues[d['label']]
        colorValues.append(colour)
        nodeLabels[n]=n
    return nodeLabels,colorValues

def getLabelsAndColour(yPred,num):
    colorValues = []
    nodeLabels={}
    for index,element in enumerate(yPred[1]):
        colour = colourValues[element[num]]
        colorValues.append(colour)
        nodeLabels["N"+str(index)]="N"+str(index)
    return nodeLabels,colorValues

def printGraph(myGraph,nodeLabels, colorValues,textEnabled,outputFile,titleText):

    pos = nx.spring_layout(myGraph)
    edge_labels=dict([((u,v,),d['weight'])for u,v,d in myGraph.edges(data=True)])

    nx.draw_networkx_nodes(myGraph, pos,node_color = colorValues)
    nx.draw_networkx_edges(myGraph, pos, arrows=False)

    if (textEnabled):
        nx.draw_networkx_labels(myGraph,pos,nodeLabels)
        nx.draw_networkx_edge_labels(myGraph,pos,edge_labels=edge_labels)

    plt.axis('off')
    plt.title(titleText)
    plt.savefig(outputFile)
    plt.show()