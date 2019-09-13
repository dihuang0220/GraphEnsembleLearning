'''
RV coefficient of two matrix, Distance Covariance, Distance coefficient of two matrix
'''

import os
import sys
import hoggorm as ho
import pandas as pd
import numpy as np
import pickle
from time import time
import networkx as nx


try: import cPickle as pickle
except: import pickle
from sklearn import model_selection as sk_ms
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import f1_score
import numpy as np
import pdb


############################################### RV coefficient for Embedding###############
t1 = time()
graphs = ['ppi','citeseer','wikipedia']
methods = ['lap','gf','hope','sdne','node2vec']
dims = [128]

path1 = 'ensemble/'
path2 = 'Coefficient/'

proc = 'final' ## or 'center'; or 'stand'
### get RV coefficiet for each method, each dimension, embeddings
for graph in graphs:
    for dim in dims:
        embeddings = []
        for i in range(len(methods)):
            e = pickle.load(open(path1+graph+"_"+methods[i]+"_"+str(dim)+".emb", "rb"))
            if proc == 'center':
                E = ho.center(e, axis=0)
            elif proc == 'stand':
                E = ho.standardise(e, mode=0)
            else:
                E = e
            embeddings+=[E]
        rv_results_cent = ho.RVcoeff(embeddings)
        print(graph, dim)
        print(rv_results_cent)
        with open(path2+graph+"_"+str(dim)+'_embedding_stand_rv.pickle','wb') as f:  pickle.dump(rv_results_cent,f)
        print('Training time: %f' % (time() - t1))



############################################### RV coefficient for prediction###############
### get RV coefficient for best combanition, embeddings
path3 = 'data/'

def get_lcc(di_graph):
    di_graph = max(nx.weakly_connected_component_subgraphs(di_graph), key=len)
    tdl_nodes = di_graph.nodes()
    nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
    di_graph = nx.relabel_nodes(di_graph, nodeListMap, copy=True)
    return di_graph, nodeListMap

class TopKRanker(oneVr):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, label] = 1
        return prediction

def evaluateNodeClassification(X_train, X_test, Y_train,Y_test):
    try:
        top_k_list = list(Y_test.toarray().sum(axis=1))
    except:
        top_k_list = list(Y_test.sum(axis=1))
    classif2 = TopKRanker(lr())
    try:
        classif2.fit(X_train, Y_train)
        prediction = classif2.predict(X_test, top_k_list)
    except:
        print('Could not fit node classification model')
        prediction = np.zeros(Y_test.shape)
    micro = f1_score(Y_test, prediction, average='micro')
    macro = f1_score(Y_test, prediction, average='macro')
    return prediction


for graph in graphs:
    for dim in dims:
        G = nx.read_gpickle(path3+graph+'/graph.gpickle')
        node_labels = pickle.load(open(path3+graph+'/node_labels.pickle', 'rb'), encoding = "latin1")
        G = G.to_undirected().to_directed()
        G, _ = get_lcc(G)
        nx.write_edgelist(G, graph+".edgelist")
        node_labels = node_labels[sorted(list(_.keys()))]

        n = G.number_of_nodes()

        test_ratio = 0.3
        test_nodes = np.random.choice(n, int(float(test_ratio)*n))
        train_nodes = list(set(G.nodes()).difference(test_nodes))

        predictions = []
        for i in range(len(methods)):
            e = pickle.load(open(path1+graph+"_"+methods[i]+"_"+str(dim)+".emb", "rb"))

            a = evaluateNodeClassification(e[train_nodes],e[test_nodes],node_labels[train_nodes],node_labels[test_nodes])
            predictions+=[a]

        rv_results = ho.RVcoeff(predictions)
        print(graph, dim)
        print(rv_results)
        with open(path2+graph+"_"+str(dim)+'_prediction_rv.pickle','wb') as f:  pickle.dump(rv_results,f)
        print('Training time: %f' % (time() - t1))

############################################### Distance  for prediction###############
import dcor
### distance covariance
#print("a, a", dcor.distance_covariance_sqr(a, a))
#print("a, b", dcor.distance_covariance_sqr(a, b))
#print("b, b", dcor.distance_covariance_sqr(b, b))

### distance covariance , coefficient statitics
# print("a, a", dcor.distance_stats_sqr(a, a))
# print("a, b", dcor.distance_stats_sqr(a, b))
# print("b, b", dcor.distance_stats_sqr(b, b))

path3 = 'data/'
for graph in graphs:
  for dim in dims:
      G = nx.read_gpickle(path3+graph+'/graph.gpickle')
      node_labels = pickle.load(open(path3+graph+'/node_labels.pickle', 'rb'), encoding = "latin1")
      G = G.to_undirected().to_directed()
      G, _ = get_lcc(G)
      nx.write_edgelist(G, graph+".edgelist")
      node_labels = node_labels[sorted(list(_.keys()))]

      n = G.number_of_nodes()

      test_ratio = 0.3
      test_nodes = np.random.choice(n, int(float(test_ratio)*n))
      train_nodes = list(set(G.nodes()).difference(test_nodes))

      predictions = []
      l = len(methods)
      for i in range(l):
          e = pickle.load(open(path1+graph+"_"+methods[i]+"_"+str(dim)+".emb", "rb"))

          a = evaluateNodeClassification(e[train_nodes],e[test_nodes],node_labels[train_nodes],node_labels[test_nodes])
          predictions+=[a]

      #rv_results = ho.RVcoeff(predictions)
      dcor_results = np.zeros([l,l])
      for i in range(l):
          for j in range(l):
              dcor_results[i][j] = dcor.distance_stats_sqr(predictions[i], predictions[j])[1]


      print(graph, dim)
      print(dcor_results)
      with open(path2+graph+"_"+str(dim)+'_prediction_dcor.pickle','wb') as f:  pickle.dump(dcor_results,f)
      print('Training time: %f' % (time() - t1))
