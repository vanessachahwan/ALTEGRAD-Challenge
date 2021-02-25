import pandas as pd
import numpy as np
import networkx as nx
from fastnode2vec import Node2Vec, Graph


# read weighted collaboration graph
WG = nx.read_edgelist("../data/weighted_collaboration_network.edgelist", nodetype=int, data=(("weight", float),))

# read similarity graph
SG = nx.read_multiline_adjlist("../data/sim_collaboration_network.adjlist", nodetype=int)

# read weighted collaboration paper graph
WPG = nx.read_edgelist("../data/weighted_graph_paper.edgelist", nodetype=int, data=(("weight", float),))

# read collaboration paper graph
PG = nx.read_edgelist("../data/graph_paper.edgelist", nodetype=int, data=(("weight", float),))

# node2vec WG
edges_list = [(str(e[0]), str(e[1]), WG[e[0]][e[1]]["weight"]) for e in WG.edges]
g = Graph(edges_list, directed=False, weighted=True)
node2vec = Node2Vec(g, dim=64, walk_length=15, context=10, p=1, q=0.5, workers=10)
node2vec.train(epochs=100)
node2vec.wv.save_word2vec_format('../data/node2vec_wg.nodevectors')

# node2vec SG
edges_list = [(str(e[0]), str(e[1]), SG[e[0]][e[1]]["weight"]) for e in SG.edges]
g = Graph(edges_list, directed=False, weighted=True)
node2vec = Node2Vec(g, dim=64, walk_length=15, context=10, p=1, q=0.5, workers=10)
node2vec.train(epochs=100)
node2vec.wv.save_word2vec_format('../data/node2vec_sg.nodevectors')

# node2vec PG
edges_list = [(str(e[0]), str(e[1]), PG[e[0]][e[1]]["weight"]) for e in PG.edges]
g = Graph(edges_list, directed=False, weighted=True)
node2vec = Node2Vec(g, dim=64, walk_length=15, context=10, p=1, q=0.5, workers=10)
node2vec.train(epochs=100)
node2vec.wv.save_word2vec_format('../data/node2vec_pg.nodevectors')

# node2vec WPG
edges_list = [(str(e[0]), str(e[1]), WPG[e[0]][e[1]]["weight"]) for e in WPG.edges]
g = Graph(edges_list, directed=False, weighted=True)
node2vec = Node2Vec(g, dim=64, walk_length=15, context=10, p=1, q=0.5, workers=10)
node2vec.train(epochs=100)
node2vec.wv.save_word2vec_format('../data/node2vec_wpg.nodevectors')
