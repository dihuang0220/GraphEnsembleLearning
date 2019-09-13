
'''
F1 score for each class
'''
import networkx as nx
import  pickle
from argparse import ArgumentParser
from scipy.io import mmread, mmwrite, loadmat
import os
import sys
import numpy as np
import pandas as pd
import json
from sklearn import model_selection as sk_ms
import itertools
from itertools import combinations as cb

from sklearn import model_selection as sk_ms
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import f1_score
from sklearn.utils import shuffle


### can add embedding combination for additional dataset 
data_method = {"ppi": {'em':[['node2vec', 'hope', 'gf', 'lap'], [64, 32,128,  64]],
						'si':[['node2vec'],[128]]},
			   "blogcat": {'em':[['node2vec', 'sdne', 'gf', 'lap'],  [128, 32, 128, 128]],
			   				'si':[['node2vec'],[128]]},
			   "citeseer": {'em':[['node2vec','sdne', 'hope', 'gf', 'lap'], [128, 128, 128, 32, 32]],
			   				'si':[['node2vec'],[128]]},
			   "wikipedia": {'em':[['hope', 'node2vec', 'sdne', 'gf', 'lap'],[128, 64, 128, 128, 64]],
			   				'si':[['hope'],[64]]}
			   }


def get_lcc(di_graph):
    di_graph = max(nx.weakly_connected_component_subgraphs(di_graph), key=len)
    tdl_nodes = di_graph.nodes()
    nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
    di_graph = nx.relabel_nodes(di_graph, nodeListMap, copy=True)
    return di_graph, nodeListMap

def get_comb_embedding(methods, dims, graph_name):
    Y = None
    for i in range(len(methods)):
        m = methods[i]
        dim = dims[i]

        m_summ = '%s_%d' % (m, dim)
        res_pre = "gem/results/ensemble/%s" % graph_name
        X = pickle.load(open('%s_%s.emb' % (res_pre, m_summ), 'rb'))
        if Y is not None:
            Y = np.concatenate((Y,X), axis=1)
        else:
            Y = X
    return Y


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
    return prediction, top_k_list, micro, macro



if __name__ == '__main__':
    ''' 
    calculate F1 score for each class in node classification
    '''
    parser = ArgumentParser(description='Efficient Graph Ensemble Experiments')
    parser.add_argument('-data', '--graph', help='graph name')
    parser.add_argument('-exp', '--experiment', help='expriment type')
    parser.add_argument('-dims', '--dimensions', help='a list of dimensions')
    parser.add_argument('-meths', '--methods', help='a list of dimensions')
    parser.add_argument('-test_ra', '--test_ratio', help='test ratio')
    parser.add_argument('-vali_ra', '--validation_ratio', help='validation ratio in the training data')
    parser.add_argument('-rounds_in', '--rounds_in', help='rounds in validation')
    parser.add_argument('-rounds_out', '--rounds_out', help='rounds in testing')
    parser.add_argument('-id', '--class_id', help='class id')
    
    params = json.load(open('ens_params.conf', 'r'))
    args = vars(parser.parse_args())
    for k, v in args.items():
        if v is not None:
            params[k] = v
    params["dimensions"].sort()

    ## load dataset
    G = nx.read_gpickle('gem/data/'+params['graph']+'/graph.gpickle')
    print('Dataset: '+ params['graph'])
    print(nx.info(G))
    node_labels = pickle.load(open('gem/data/'+params['graph']+'/node_labels.pickle', 'rb'), encoding = "latin1")
    G = G.to_undirected().to_directed()
    G, _ = get_lcc(G)
    nx.write_edgelist(G, params["graph"]+".edgelist")
    node_labels = node_labels[sorted(list(_.keys()))]
    print('Dataset: '+ params['graph'])
    print(nx.info(G))

    n = G.number_of_nodes()
    test_nodes = np.random.choice(n, int(float(params["test_ratio"][0])*n))
    train_nodes = list(set(np.arange(n)).difference(test_nodes))

    predictions = []
    embed = get_comb_embedding(data_method[params['graph']]['em'][0], data_method[params['graph']]['em'][1], params['graph'])
    prediction,top_k, mi, ma = evaluateNodeClassification(embed[train_nodes],\
                                embed[test_nodes],node_labels[train_nodes],node_labels[test_nodes])
    predictions += [prediction]

    embed = get_comb_embedding(data_method[params['graph']]['si'][0], data_method[params['graph']]['si'][1], params['graph'])
    prediction,top_k, mi, ma = evaluateNodeClassification(embed[train_nodes],\
                                embed[test_nodes],node_labels[train_nodes],node_labels[test_nodes])
    predictions += [prediction]

    Y = node_labels[test_nodes].toarray()

    f1_result = []
    for i in range(len(Y[0])):
    	n1 = np.sum(Y[:,i])
    	n2 = np.sum(node_labels.toarray()[:,i])
    	f1_1 = f1_score(Y[:,i], predictions[0][:,i], average = 'binary')
    	f1_2 = f1_score(Y[:,i], predictions[1][:,i], average = 'binary')
    	f1_result += [[n1, n2, f1_1, f1_2]]

    pickle.dump(f1_result, open("./"+ params['graph']+"_minority.pickle", 'wb'))

























