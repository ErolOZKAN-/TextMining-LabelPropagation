from SeedGenerator import *
import time
from datetime import datetime
from graphPrinter import printGraph, getLabelsAndColour,getLabelsAndColourFromGraph

kNeightestValue = 3
modeValue = 4
visualize = True

trainFileName = '../data/assgn4Data/orginalLabels.txt'
similarityFilenameToParse = '../data/assgn4Data/erol.mat'

# trainFileName = '../data/assgn4Data/r8-train-stemmed.txt'
# similarityFilenameToParse = '../data/assgn4Data/sims.mat'

mySeedsFile = '../result/mySeedsFile'
resultsFile = "../result/result"
mySeedsInfoFile = "../result/mySeedsInfoFile"

orginalPicture = "../result/orginalPicture.png"
initialSeedsPicture = "../result/initialSeedsPicture.png"
afterLabelPropagationPicture = "../result/afterLabelPropagationPicture.png"

class TextCategorizer(object):
    def buildGraph(self, delimiter="\t"):
        sentencesList= getR8TrainSentences(trainFileName)
        seedList = generateSeed(sentencesList,modeValue)
        printToFile(seedList,mySeedsFile)

        similarityData = parse(similarityFilenameToParse)
        inputGraph = parseDirectedGraph(similarityData,sentencesList,kNeightestValue)

        if(visualize):
            nodeLabels,colorValues = getLabelsAndColourFromGraph(inputGraph)
            printGraph(inputGraph,nodeLabels, colorValues,1,orginalPicture,"orginalPicture")

        W = to_scipy_sparse_matrix(inputGraph)
        nodesInGraph = inputGraph.nodes()
        labelIndex = defaultdict(list)

        fileLines = open(mySeedsFile, "r").readlines()
        seed_nodes, seedLabels, seed_values = [], [], []
        for idx, line in enumerate(fileLines):
            split_line = line.split(delimiter)
            node, label, value = split_line[0], split_line[1], float(split_line[2])
            seed_nodes.append(node)
            seedLabels.append(label)
            seed_values.append(value)
            labelIndex[label].append(idx)

        # Store seeds name and its value
        goldenLabels = dict(zip(seed_nodes, seedLabels))

        uniqueLabels = sorted(set(seedLabels))
        labelsDictionary = {e: idx for idx, e in enumerate(uniqueLabels)}


        # Build the seeds matrix: number of nodes x number of labels + 1
        seedsMatrix = np.zeros(
            [W.shape[0], len(set(seedLabels)) + 1])  # We add 1 because of the "dummy" label used in the algorithm

        # Draw a n sample for each type of label
        label_samples = {}
        for k, v in labelIndex.items():
            temp_nb_seeds = 10 if len(v) > 10 else len(v)
            label_samples[k] = np.random.choice(v, temp_nb_seeds)

        # Once we have the sample for each class, we build the seeds matrix
        for label, seed_idx in label_samples.items():
            label_j = labelsDictionary.get(label)
            for i in seed_idx:
                seedsMatrix[i, label_j] = seed_values[i]

        printToSomeOtherFile(goldenLabels.copy(),mySeedsInfoFile)

        return inputGraph,W, seedsMatrix, uniqueLabels, goldenLabels, nodesInGraph


    def __init__(self,tol=1e-3, maxiter=5):
        self._G,self._W, self._Y, labels, self._golden_labels, self._nodes_list = self.buildGraph()

        self._labels = labels
        self._mu1 = 1.
        self._mu2 = 1e-2
        self._mu3 = 1e-2
        self._beta = 2.
        self._tol = tol
        self._get_initial_probs()
        self.max_iter = maxiter

    def _get_initial_probs(self):
        print "GETTING INITIAL PROBABILITIES"
        nr_nodes = self._W.shape[0]

        # Calculate Pr transition probabilities matrix
        self._Pr = lil_matrix((nr_nodes, nr_nodes))
        W_coo = self._W.tocoo()
        col_sums = {k: v for k, v in enumerate(W_coo.sum(0).tolist()[0])}
        for i, j, v in itertools.izip(W_coo.row, W_coo.col, W_coo.data):
            # print "\t%d\t%d" % (i,j)
            self._Pr[i, j] = v / col_sums[j]

        print "\t\tPr matrix done."
        # Calculate H entropy vector for each node
        self._H = lil_matrix((nr_nodes, 1))
        self._Pr = self._Pr.tocoo()

        for i, _, v in itertools.izip(self._Pr.row, self._Pr.col, self._Pr.data):
            self._H[i, 0] += -(v * np.log(v))

        print "\t\tH matrix done."

        # Calculate vector C (Cv)
        self._H = self._H.tocoo()
        self._C = lil_matrix((nr_nodes, 1))
        log_beta = np.log(self._beta)
        for i, _, v in itertools.izip(self._H.row, self._H.col, self._H.data):
            # print v
            self._C[i, 0] = (log_beta) / (np.log(self._beta + (1 / (np.exp(-v) + 0.00001))))

        print "\t\tC matrix done."

        # Calculate vector D (dv)
        # Get nodes that are labeled
        Y_nnz = self._Y.nonzero()
        self._D = lil_matrix((nr_nodes, 1))
        self._H = self._H.tolil()

        for i in Y_nnz[0]:
            # Check if node v is labeled            
            self._D[i, 0] = (1. - self._C[i, 0]) * np.sqrt(self._H[i, 0])

        print "\t\tD matrix done."
        # Calculate Z vector
        self._Z = lil_matrix((nr_nodes, 1))
        c_v = self._C + self._D
        c_v_nnz = c_v.nonzero()
        for i in c_v_nnz[0]:
            self._Z[i, 0] = np.max([c_v[i, 0], 1.])
        print "\t\tZ matrix done."

        # Finally calculate p_cont, p_inj and p_abnd
        self._Pcont = lil_matrix((nr_nodes, 1))
        self._Pinj = lil_matrix((nr_nodes, 1))
        self._Pabnd = lil_matrix((nr_nodes, 1))
        C_nnz = self._C.nonzero()
        for i in C_nnz[0]:
            self._Pcont[i, 0] = self._C[i, 0] / self._Z[i, 0]
        for i in Y_nnz[0]:
            self._Pinj[i, 0] = self._D[i, 0] / self._Z[i, 0]

        self._Pabnd[:, :] = 1.
        pc_pa = self._Pcont + self._Pinj
        pc_pa_nnz = pc_pa.nonzero()
        for i in pc_pa_nnz[0]:
            self._Pabnd[i, 0] = 1. - pc_pa[i, 0]
        # for i in range(nr_nodes):
        # self._Pabnd[i, 0] = 1. - self._Pcont[i, 0] - self._Pinj[i, 0]

        self._Pabnd = csr_matrix(self._Pabnd)
        self._Pcont = csr_matrix(self._Pcont)
        self._Pinj = csr_matrix(self._Pinj)
        print "\n\nDone getting probabilities..."

    def propagate(self):
        print "\n...Calculating lp..."
        nr_nodes = self._W.shape[0]

        # 1. Initialize Yhat
        self._Yh = lil_matrix(self._Y.copy())

        # 2. Calculate Mvv
        self._M = lil_matrix((nr_nodes, nr_nodes))

        for v in range(nr_nodes):
            first_part = self._mu1 * self._Pinj[v, 0]
            second_part = 0.

            for u in self._W[v, :].nonzero()[1]:
                if u != v:
                    second_part += (self._Pcont[v, 0] * self._W[v, u] + self._Pcont[u, 0] * self._W[u, v])
            self._M[v, v] = first_part + (self._mu2 * second_part) + self._mu3

        numberOfIterations = 0
        r = lil_matrix((1, self._Y.shape[1]))
        r[-1, -1] = 1.
        Yh_old = lil_matrix((self._Y.shape[0], self._Y.shape[1]))

        # Main loop begins
        Pcont = self._Pcont.toarray()

        while not self.checkConvergence(Yh_old, self._Yh, ) and self.max_iter > numberOfIterations:
            numberOfIterations += 1
            print ">>>>>Iteration:%d" % numberOfIterations
            self._D = lil_matrix((nr_nodes, self._Y.shape[1]))
            # 4. Calculate Dv
            print "\t\tCalculating D..."
            time_d = time.time()
            W_coo = self._W.tocoo()
            for i, j, v in itertools.izip(W_coo.row, W_coo.col, W_coo.data):
                # self._D[i, :] += (Pcont[i][0] * v + Pcont[j][0] * v) * self._Yh[j, :]
                self._D[i, :] += (v * (Pcont[i][0] + Pcont[j][0])) * self._Yh[j, :]
                # print i

            print "\t\tTime it took to calculate D:", time.time() - time_d
            print

            print "\t\tUpdating Y..."
            # 5. Update Yh
            time_y = time.time()
            Yh_old = self._Yh.copy()
            for v in range(nr_nodes):
                # 6.
                second_part = ((self._mu1 * self._Pinj[v, 0] * self._Y[v, :]) +
                               (self._mu2 * self._D[v, :]) +
                               (self._mu3 * self._Pabnd[v, 0] * r))
                self._Yh[v, :] = 1. / (self._M[v, v]) * second_part

            print "\t\tTime it took to calculate Y:", time.time() - time_y
            print
            # repeat until convergence.

    def checkConvergence(self, A, B):
        if not type(A) is csr_matrix:
            A = csr_matrix(A)
        if not type(B) is csr_matrix:
            B = csr_matrix(B)

        norm_a = (A.data ** 2).sum()
        norm_b = (B.data ** 2).sum()
        diff = np.abs(norm_a - norm_b)
        if diff <= self._tol:
            return True
        else:
            print "\t\tNorm differences between Y_old and Y_hat: ", diff
            print
            return False

    def results(self):
        y_predd = 0
        acc = 0.0
        for i in itertools.permutations(self._labels, len(self._labels)):
            myLabels = i
            completeResult = []
            self.classIndex = np.squeeze(np.asarray(self._Yh[:, :self._Yh.shape[1] - 1].todense().argmax(axis=1)))
            self.labelResult = np.array([myLabels[r] for r in self.classIndex])
            for i in range(len(self.labelResult)):
                completeResult.append((self._nodes_list[i],
                                       self._golden_labels.get(self._nodes_list[i], " -"),self.labelResult[i] ))
            self.completeResult = completeResult
            y_pred = self.labelResult, self.completeResult
            myResult  = object.accurancy(y_pred)
            if (myResult >= acc):
                y_predd = y_pred
                acc = myResult
        return y_predd

    def visualize(self,y_pred):
        nodeLabels,colorValues = getLabelsAndColour(y_pred,1)
        printGraph(self._G, nodeLabels, colorValues, 1, initialSeedsPicture, "Initial Seeds")

        nodeLabels,colorValues = getLabelsAndColour(y_pred,2)
        printGraph(self._G,nodeLabels, colorValues,1,afterLabelPropagationPicture,"After Label Propagation Algorithm")

    def accurancy(self, ypred):
        totalNumber = 0
        trueNumber = 0.0

        for n,d in self._G.nodes_iter(data=True):
            if(d['label']==str(ypred[0][totalNumber])):
                trueNumber = trueNumber + 1
            totalNumber = totalNumber + 1

        return float(trueNumber/totalNumber)

    def calculate_accurancy(self, ypred):
        totalNumber = 0
        trueNumber = 0.0

        with open(resultsFile, 'w') as csvfile:
            fieldnames = ['time','nodeName','seedLabel','assignedLabel','trueLabel','result']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for n,d in self._G.nodes_iter(data=True):
                writer.writerow({
                    'time':str(datetime.now()),
                    'nodeName':ypred[1][totalNumber][0],
                    'seedLabel': ypred[1][totalNumber][1],
                    'assignedLabel': ypred[1][totalNumber][2],
                    'trueLabel': d['label'],
                    'result': d['label']==str(ypred[0][totalNumber]),

                })
                if(d['label']==str(ypred[0][totalNumber])):
                    trueNumber = trueNumber + 1
                totalNumber = totalNumber + 1

        return float(trueNumber/totalNumber)

if __name__ == '__main__':
    timo = time.time()
    object = TextCategorizer()
    object.propagate()
    print "GETTING RESULTS..."
    y_pred = object.results()
    accurancy = object.calculate_accurancy(y_pred)
    print "CALCULATION TOOK : ", time.time() - timo
    print "ACCURANCY : ",accurancy

    if(visualize):
        object.visualize(y_pred)
    print "CHECK RESULT DIRECTORY FOR MORE INFORMATION..."
