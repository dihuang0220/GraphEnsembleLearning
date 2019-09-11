'''
Efficient Graph Ensemble: BC-baseline concatenation
'''

from time import time
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


from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.evaluation import evaluate_node_classification as e_nc
from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne     import SDNE



methClassMap = {"gf": "GraphFactorization",
                "hope": "HOPE",
                "lle": "Locally Linear Embedding",
                "lap": "LaplacianEigenmaps",
                "node2vec": "node2vec",
                "sdne": "SDNE"}


def convert_nx_graph(G):
    edge_list = []
    for k,v in G.items():
        for i in v:
            edge_list.append((int(k), int(i)))
    g = nx.DiGraph()
    g.add_edges_from(edge_list)
    return g


def get_lcc(di_graph):
    di_graph = max(nx.weakly_connected_component_subgraphs(di_graph), key=len)
    tdl_nodes = di_graph.nodes()
    nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
    di_graph = nx.relabel_nodes(di_graph, nodeListMap, copy=True)
    return di_graph, nodeListMap


def get_max(val, val_max, idx, idx_max):
    if val > val_max:
        return val, idx
    else:
        return val_max, idx_max

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


if __name__ == '__main__':
    ''' 
    Efficient graph ensemble on Node Classification Task
    --rounds_in: rounds on evaluation on validation. 
    --rounds_out: rounds on evalution on test. use the embedding from search for best hyp. 
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
    

    params = json.load(open('Graph_Ensemble/ens_params.conf', 'r'))
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

    try:
        os.makedirs("gem/experiments/config/ensemble/")
    except:
        pass

    try:
        os.makedirs("gem/results/ensemble/")
    except:
        pass

    try:
        os.makedirs("gem/intermediate/ensemble/")
    except:
        pass

    try:
        os.makedirs("gem/results/ensemble_test/")
    except:
        pass


    ######## BASELINE BC######
    if params['experiment'] == "baseline":
        ## split dataset into train, validation, test
        n = G.number_of_nodes()
        test_nodes = np.random.choice(n, int(float(params["test_ratio"][0])*n))
        train_nodes = list(set(G.nodes()).difference(test_nodes))

        method_best_dim = {m:0 for m in params['methods']}
        nc_method_best_dim = {m:0 for m in params['methods']}
        for dim in params['dimensions']:
            for method in params['methods']:
                best_X = None
                try:
                    model_hyp_range = json.load(
                    open('gem/experiments/config/%s_hypRange.conf' % params['graph'], 'r')
                    )
                except IOError:
                    model_hyp_range = json.load(
                    open('gem/experiments/config/default_hypRange.conf', 'r')
                    )
                MethClass = getattr(
                    importlib.import_module("gem.embedding.%s" % method),
                    methClassMap[method])
                meth_hyp_range = model_hyp_range[method]
                nc_max = 0
                nc_hyp = {method: {}}
                n_r = params["rounds_in"]
                ev_cols = ["NC F1 score"]
                hyp_df = pd.DataFrame(
                columns=list(meth_hyp_range.keys()) + ev_cols + ["Round Id"]
                )
                hyp_r_idx = 0
                for hyp in itertools.product(*meth_hyp_range.values()):
                    hyp_d = {"d": dim}
                    hyp_d.update(dict(zip(meth_hyp_range.keys(), hyp)))
                    print(hyp_d)
                    if method == "sdne":
                        hyp_d.update({
                            "modelfile": [
                                "gem/intermediate/enc_mdl_%s_%d.json" % (params['graph'], dim),
                                "gem/intermediate/dec_mdl_%s_%d.json" % (params['graph'], dim)
                            ],
                            "weightfile": [
                                "gem/intermediate/enc_wts_%s_%d.hdf5" % (params['graph'], dim),
                                "gem/intermediate/dec_wts_%s_%d.hdf5" % (params['graph'], dim)
                            ]
                        })
                    elif method == "gf" or method == "node2vec":
                        hyp_d.update({"data_set": params['graph']})
                    MethObj = MethClass(hyp_d)
                    m_summ = '%s_%d' % (method, dim)
                    res_pre = "gem/results/ensemble/%s" % params['graph']

                    print('Learning Embedding: %s' % m_summ)

                    X, learn_t = MethObj.learn_embedding(graph=G,
                                                    is_weighted=False,
                                                    edge_f=None,
                                                    no_python=True)
                    print('\tTime to learn embedding: %f sec' % learn_t)
                    
                    nc = [0] * n_r
                    nc = e_nc.expNC(X[train_nodes], 
                               node_labels[train_nodes],
                               params["validation_ratio"],
                               n_r, res_pre, m_summ)
                    print("nc", nc)
                    nc_m = np.mean(nc)
                    if nc_m >= nc_max:
                        best_X = X
                    nc_max, nc_hyp[method] = get_max(nc_m, nc_max, hyp_d, nc_hyp[method])
                    hyp_df_row = dict(zip(meth_hyp_range.keys(), hyp))
                    ## record each nc result
                    for r_id in range(n_r):
                        hyp_df.loc[hyp_r_idx, meth_hyp_range.keys()] = pd.Series(hyp_df_row)
                        hyp_df.loc[hyp_r_idx, ev_cols + ["Round Id"]] = [nc[r_id], r_id]
                        hyp_r_idx += 1
                hyp_df.to_hdf(
                    "gem/intermediate/ensemble/%s_%s_%s_%s_hyp.h5" % (params['graph'], method,params['experiment'],dim), "df")
                opt_hyp_f_pre = 'gem/experiments/config/ensemble/%s_%s_%s' % (params['graph'],method,dim)

                if nc_max and best_X is not None:
                    with open('%s_nc.conf' % opt_hyp_f_pre, 'w') as f:
                        f.write(json.dumps(nc_hyp, indent=4))
                    pickle.dump(best_X, open('%s_%s.emb' % (res_pre, m_summ), 'wb'))

                ##get best dim for each method
                if nc_max>=nc_method_best_dim[method]:
                    method_best_dim[method] = dim
                    nc_method_best_dim[method] = nc_max

        print('method_best_dim : ', method_best_dim)
        print('nc_method_best_dim :', nc_method_best_dim)
        with open("gem/intermediate/ensemble/%s_%s_methodbest_dim.pickle" % (params['graph'],params['experiment']), 'wb') as fp:
            pickle.dump(method_best_dim, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open("gem/intermediate/ensemble/%s_%s_methodbest_nc.pickle" % (params['graph'],params['experiment']), 'wb') as fp:
            pickle.dump(nc_method_best_dim, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open("gem/intermediate/ensemble/%s_%s_methodbest_dim.pickle" % (params['graph'],params['experiment']), 'rb') as fp:
            method_best_dim = pickle.load(fp)
        with open("gem/intermediate/ensemble/%s_%s_methodbest_nc.pickle" % (params['graph'],params['experiment']), 'rb') as fp:
            nc_method_best_dim = pickle.load(fp)

        n_r = params["rounds_in"]
        nc_max = 0
        comb_max = []

        best_comb_df = pd.DataFrame(
                 columns= ["combination","dim","NC F1 score"])
        idx = 0
        ms = list(method_best_dim.keys())
        Y = get_comb_embedding(ms, [method_best_dim[m] for m in ms], params['graph'])
        Y_all_best_comb = Y
        nc = [0] * n_r
        nc = e_nc.expNC(Y[train_nodes], 
                         node_labels[train_nodes],
                         params["validation_ratio"],
                         n_r, "gem/results/ensemble/%s" % params['graph'], "alL_best_comb" )
        nc_m = np.mean(nc)
        best_comb_df.loc[idx] = [("all best combine"),0,nc_m]
        idx+=1
        nc_max, comb_max = get_max(nc_m, nc_max, ["all best combine"], comb_max)

        for c in range(2,len(params['methods'])+1):
            for comb in cb(params['methods'], c):
                for dim_com in itertools.product(*[params['dimensions']]*len(comb)):
                    Y = get_comb_embedding(comb, dim_com, params['graph'])
                    nc = [0] * n_r
                    nc = e_nc.expNC(Y[train_nodes], 
                         node_labels[train_nodes],
                         params["validation_ratio"],
                         n_r, "gem/results/ensemble/%s" % params['graph'], '_'.join(comb) + '_'.join([str(i) for i in dim_com]))
                    nc_m = np.mean(nc)
                    best_comb_df.loc[idx] = [comb,dim_com,nc_m]
                    idx+=1
                    nc_max, comb_max = get_max(nc_m, nc_max, [comb, dim_com], comb_max)

        best_comb_df.to_hdf(
                 "gem/intermediate/ensemble/%s_%s_allcomb_hyp.h5" % (params['graph'],params['experiment']), "df")              

        baseline_df = pd.DataFrame(
                 columns= ["methods","dim","NC F1 score"])
        idx = 0
        n_r = params['rounds_out']
        for m in method_best_dim.keys():
            Y = get_comb_embedding([m], [method_best_dim[m]], params['graph'])
            nc = e_nc.expNC(Y, 
                         node_labels,
                         params['test_ratio'],
                         n_r,  "gem/results/ensemble_test/%s" % params['graph'], '%s_%d' % (m, method_best_dim[m]))
            nc_m = np.mean(nc)
            baseline_df.loc[idx] = [m, method_best_dim[m],nc_m]
            idx+=1
        ### best comb
        if comb_max == ["all best combine"]:
            Y = Y_all_best_comb
        else:
            Y = get_comb_embedding(comb_max[0], comb_max[1], params['graph'])
        nc = e_nc.expNC(Y, 
                     node_labels,
                     params['test_ratio'],
                     n_r,  "gem/results/ensemble_test/%s" % params['graph'], '%s' % (str(comb_max)))
        nc_m = np.mean(nc)
        baseline_df.loc[idx] = [str(comb_max), 0,nc_m]
        baseline_df.to_hdf(
                 "gem/results/ensemble_test/%s_%s_test.h5" % (params['graph'],params['experiment']), "df")

        print("Baseline Finish!")
