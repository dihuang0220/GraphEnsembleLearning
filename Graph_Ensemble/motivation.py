import networkx as nx
import os
import subprocess
if os.name == 'posix' and 'DISPLAY' not in os.environ:
	print("Using raster graphics – high quality images using the Anti-Grain Geometry (AGG) engine")
	import matplotlib
	matplotlib.use('Agg')
	matplotlib.rcParams['text.latex.unicode']=False
import matplotlib.pyplot as plt

import seaborn
from matplotlib import rc
import numpy as np
import pdb
import random
import json
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csc_matrix
import pickle

# get the modules for generating the synthetic graphs
import sys
sys.path.insert(0, '../')
from gem.utils import graph_util, graph_gens
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
                "lap": "LaplacianEigenmaps",
                "node2vec": "node2vec",
                "sdne": "SDNE",
                "pa": "PreferentialAttachment",
                "rand": "RandomEmb",
                "cn": "CommonNeighbors",
                "aa": "AdamicAdar",
                "jc": "JaccardCoefficient"}

# set the searborn and matplotlib formatting style
font = {'family': 'serif', 'serif': ['computer modern roman']}
rc('text', usetex=True)
rc('font', weight='bold')
rc('font', size=20)
rc('lines', markersize=10)
rc('xtick', labelsize=5)
rc('ytick', labelsize=5)
rc('axes', labelsize='x-large')
rc('axes', labelweight='bold')
rc('axes', titlesize='x-small')
rc('axes', linewidth=3)

plt.rc('font', **font)
seaborn.set_style("darkgrid")

cmap = plt.cm.get_cmap('Set1')
title = 'synthetic'


if os.name == 'posix':
	os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/Root/bin/x86_64-darwin'

def get_node_color(labels):
    node_colors = [cmap(c) for c in labels]
    return node_colors

def plot_embedding2D(node_pos, node_colors=None, di_graph=None, labels=None, shape = None):
    if shape is None:
    	embedding_dimension = node_pos[0].shape[0]
    else:
    	embedding_dimension = shape
    if (embedding_dimension > 2):
        print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    if di_graph is None:
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors)
    else:
        if node_colors:
            nodes_draw = nx.draw_networkx_nodes(di_graph, node_pos,
                                                node_color=node_colors,
                                                width=0.2, node_size=40,
                                                arrows=False, alpha=0.9,
                                                font_size=5)
            nodes_draw.set_edgecolor('w')
        else:
            nodes_draw = nx.draw_networkx(di_graph, pos, node_color=node_colors,
                                          width=0.2, node_size=40, arrows=False,
                                          alpha=0.9, font_size=12)
            nodes_draw.set_edgecolor('w')
        nx.draw_networkx_edges(di_graph,node_pos,arrows=False,width=0.4,alpha=0.8,edge_color='#6B6B6B') 

def expVis(X, gname='test', node_labels=None, di_graph=None, lbl_dict=None, title = 'test'):
    print('\tGraph Visualization:')
    pos =1
    for i in range(len(gname)):
        ax= plt.subplot(220 + pos)
        pos += 1
        ax.title.set_text(gname[i])
        if node_labels[i]:
           node_colors = get_node_color(node_labels[i])
        else:
           node_colors = None

        plot_embedding2D(X[i], node_colors=node_colors,
	                     di_graph=di_graph[i], labels =lbl_dict[i] )
    plt.savefig('ensemble_%s.pdf' % (title), dpi=300,
                format='pdf', bbox_inches='tight')
    plt.figure()

G_list =	[]
gname_list = []

gname_tmp = 'Barabasi Albert Graph'
G_tmp, d, dim = graph_gens.barabasi_albert_graph(100,1,0,3,'social')
G_list.append(G_tmp)
gname_list.append(gname_tmp)
gname_tmp = 'Random Geometric Graph'
G_tmp, _, _ = graph_gens.random_geometric_graph(100,5,0,3,'social')
G_list.append(G_tmp)
gname_list.append(gname_tmp)
gname_tmp = 'Stochastic Block Model Graph'
G_tmp, d, dim = graph_gens.stochastic_block_model(100,5,0,3,'social')
G_list.append(G_tmp)
gname_list.append(gname_tmp)
gname_tmp = 'Watts Strogatz Graph'
G_tmp = nx.watts_strogatz_graph(n=100, k=3, p=0.2)
G_list.append(G_tmp)
gname_list.append(gname_tmp)


def process_synthetic_graphs(G_list, gname_list):
	G  =[]
	pos =[]
	node_labels =[]
	lbl_dict =[]
	gname =[]

	for G_tmp,gname_tmp in zip(G_list,gname_list):
		pos_tmp = nx.spring_layout(G_tmp)
		nodes_deg =[G_tmp.degree[i] for i in G_tmp.nodes()]
		unq_lbl = np.unique(nodes_deg)
		lbl_map = {unq_lbl[i]:i for i in range(len(unq_lbl))}
		lbl_map_rev = {v:k for k,v in lbl_map.items()}

		node_labels_tmp = [lbl_map[k] for k in nodes_deg]
		lbl_dict_tmp = {n:i for n,i in enumerate(G_tmp.nodes())}

		G.append(G_tmp)
		pos.append(pos_tmp)
		node_labels.append(node_labels_tmp)
		lbl_dict.append(lbl_dict_tmp)
		gname.append(gname_tmp)

	return G, pos, node_labels, lbl_dict, gname
G, pos, node_labels, lbl_dict, gname = process_synthetic_graphs(G_list, gname_list)

expVis(pos, gname =gname,
	node_labels=node_labels,
	di_graph=G, 
	lbl_dict=lbl_dict,
	title = 'synthetic')
if os.name == 'posix':
	bashCommand = "open ensemble_%s.pdf" % (title)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

def get_labels(measure):
	_,bins = np.histogram(list(measure.values()),bins='auto')
	bins= {i:(bins[i],bins[i+1]) for i in range(len(bins)-1)}
	labels ={}
	for k,v in measure.items():
		for k2,v2 in bins.items():
			if v2[0]<=v<=v2[1]:
				labels[k]=k2
				continue
	return labels

def mergegraph(graphs,pos_old, labels_old, edge_prob=0.3, edge_num=0.4):
	nodes = []
	edges = []
	pos = {}
	node_cnt = 0
	val =0.9
	shift_value =[[-val,val],[val,val],[-val,-val],[val,-val]]
	
	comm_lables = []

	for i,g in enumerate(graphs):
		tmp_nodes = list(g.nodes())
		tmp_edges = list(g.edges())

		comm_lables+=[i]*len(tmp_nodes)

		node_map = {k:node_cnt+i for k,i in enumerate(tmp_nodes)}
		node_cnt+=len(tmp_nodes)

		new_nodes = [node_map[n] for n in tmp_nodes]
		new_edges = [(node_map[u],node_map[v]) for u,v in tmp_edges]

		for k,v in pos_old[i].items():
			pos_old[i][k][0]+=shift_value[i][0]
			pos_old[i][k][1]+=shift_value[i][1]

		new_pos = {node_map[n]:v for n,v in pos_old[i].items()}


		nodes+=new_nodes
		edges+=new_edges
		pos.update(new_pos)


	G = nx.DiGraph()
	G.add_edges_from(edges)

	random.shuffle(nodes)
	l = int(edge_num*len(nodes))
	u = nodes[0:l]
	random.shuffle(nodes)
	v = nodes[0:l]

	for s, t in zip(u,v):
		if random.random()<edge_prob:
			G.add_edge(s,t)
			G.add_edge(t,s)
	nodes_deg =[G.degree[i] for i in G.nodes()]
	
	centrality = nx.closeness_centrality(G)
	labels_central = get_labels(centrality)
	print('centrality done!')
	
	inf_cent=nx.information_centrality(G.to_undirected())
	labels_inf_central = get_labels(inf_cent)
	print('info centrality done!')


	betweenness = nx.betweenness_centrality(G.to_undirected())
	labels_betweenness= get_labels(betweenness)
	print('betweenness done!')

	loads = nx.load_centrality(G.to_undirected())
	labels_load = get_labels(loads)
	print('load centrality done!')


	cmm_bet=nx.communicability_betweenness_centrality(G.to_undirected())
	labels_cmm_bet = get_labels(cmm_bet)
	print('commu betweenness done!')

	sce = nx.subgraph_centrality_exp(G.to_undirected())
	labels_sce = get_labels(sce)
	print('subgraph centrality done!')


	harm = nx.harmonic_centrality(G.to_undirected())
	labels_harm = get_labels(harm)
	print('harmonic done!')

	lrc={v:nx.local_reaching_centrality(G.to_undirected(),v) for v in G.nodes()}
	labels_lrc = get_labels(lrc)
	print('lrc done!')
				
				
	unq_lbl = np.unique(nodes_deg)
	lbl_map = {unq_lbl[i]:i for i in range(len(unq_lbl))}
	labels = [lbl_map[k] for k in nodes_deg]
	return G, pos, labels, comm_lables, labels_central, labels_inf_central, labels_betweenness, labels_load, labels_cmm_bet, labels_sce, labels_harm, labels_lrc

G, pos, labels, comm_lables, labels_central, labels_inf_central, labels_betweenness, labels_load, labels_cmm_bet, labels_sce, labels_harm, labels_lrc=mergegraph(G, pos,node_labels)

path_1 = "../gem/data/motivationgraph_degree/"
path_2 = "../gem/data/motivationgraph_community/"
path_3 = "../gem/data/motivationgraph_centrality/"
path_4 = "../gem/data/motivationgraph_information_centrality/"
path_5 = "../gem/data/motivationgraph_betweenness/"
path_6 = "../gem/data/motivationgraph_load_centrality/"
path_7 = "../gem/data/motivationgraph_communicability_betweenness_centrality/"
path_8 = "../gem/data/motivationgraph_subgraph_centrality_exp/"
path_9 = "../gem/data/motivationgraph_harmonic_centrality/"
path_10 = "../gem/data/motivationgraph_local_reaching_centrality/"

try:
	os.makedirs(path_1)
except:
	pass
try:
	os.makedirs(path_2)
except:
	pass
try:
	os.makedirs(path_3)
except:
	pass
try:
	os.makedirs(path_4)
except:
	pass
try:
	os.makedirs(path_5)
except:
	pass
try:
	os.makedirs(path_6)
except:
	pass
try:
	os.makedirs(path_7)
except:
	pass
try:
	os.makedirs(path_8)
except:
	pass
try:
	os.makedirs(path_9)
except:
	pass
try:
	os.makedirs(path_10)
except:
	pass

nx.write_gpickle(G, path_1+"graph.gpickle")
nx.write_gpickle(G, path_2+"graph.gpickle")
nx.write_gpickle(G, path_3+"graph.gpickle")
nx.write_gpickle(G, path_4+"graph.gpickle")
nx.write_gpickle(G, path_5+"graph.gpickle")
nx.write_gpickle(G, path_6+"graph.gpickle")
nx.write_gpickle(G, path_7+"graph.gpickle")
nx.write_gpickle(G, path_8+"graph.gpickle")
nx.write_gpickle(G, path_9+"graph.gpickle")
nx.write_gpickle(G, path_10+"graph.gpickle")

def onehot(labels):
	onehot_encoder = OneHotEncoder(sparse=False)
	labels = np.array(labels)
	labels = labels.reshape(len(labels),1)
	labels = onehot_encoder.fit_transform(labels)
	labels = csc_matrix(labels)
	return labels

def turn_list(d):
	return [d[i] for i in range(len(d))]

pickle.dump(onehot(labels), open(path_1+'node_labels.pickle', 'wb'))
pickle.dump(onehot(comm_lables), open(path_2+'node_labels.pickle', 'wb'))
pickle.dump(onehot(turn_list(labels_central)), open(path_3+'node_labels.pickle', 'wb'))
pickle.dump(onehot(turn_list(labels_inf_central)), open(path_4+'node_labels.pickle', 'wb'))
pickle.dump(onehot(turn_list(labels_betweenness)), open(path_5+'node_labels.pickle', 'wb'))
pickle.dump(onehot(turn_list(labels_load)), open(path_6+'node_labels.pickle', 'wb'))
pickle.dump(onehot(turn_list(labels_cmm_bet)), open(path_7+'node_labels.pickle', 'wb'))
pickle.dump(onehot(turn_list(labels_sce)), open(path_8+'node_labels.pickle', 'wb'))
pickle.dump(onehot(turn_list(labels_harm)), open(path_9+'node_labels.pickle', 'wb'))
pickle.dump(onehot(turn_list(labels_lrc)), open(path_10+'node_labels.pickle', 'wb'))

colors = get_node_color(labels)

plot_embedding2D(pos, node_colors=colors, di_graph=G, labels=None, shape =2)
plt.savefig('ensemble_merged.pdf', dpi=300,
                format='pdf', bbox_inches='tight')
plt.figure()

if os.name == 'posix':
	bashCommand = "open ensemble_merged.pdf" 
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

pos = nx.spring_layout(G)
plot_embedding2D(pos, node_colors=colors, di_graph=G, labels=None, shape =2)
plt.savefig('ensemble_merged_spring.pdf', dpi=300,
                format='pdf', bbox_inches='tight')

if os.name == 'posix':
	bashCommand = "open ensemble_merged_spring.pdf" 
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()


