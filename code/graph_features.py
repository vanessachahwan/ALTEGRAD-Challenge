import pandas as pd
import numpy as np
import networkx as nx

from multiprocessing import Pool
import itertools
import pickle

# load the three graphs
G = nx.read_edgelist('../data/collaboration_network.edgelist', delimiter=' ', nodetype=int)
WG = nx.read_edgelist("../data/weighted_collaboration_network.edgelist", nodetype=int, data=(("weight", float),))
SG = nx.read_multiline_adjlist("../data/sim_collaboration_network.adjlist", nodetype=int)

def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_subset,
        zip(
            [G] * num_chunks,
            node_chunks,
            [list(G)] * num_chunks,
            [True] * num_chunks,
            [None] * num_chunks,
        ),
    )

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c

# compute the betweenness centrality for all the graphs
betweenness_centrality_g = betweenness_centrality_parallel(G, 50)
betweenness_centrality_wg = betweenness_centrality_parallel(WG, 50)
betweenness_centrality_sg = betweenness_centrality_parallel(SG, 50)

f = open("../data/betweenness_centrality_g.pkl","wb")
pickle.dump(betweenness_centrality_g,f)
f.close()

f = open("../data/betweenness_centrality_wg.pkl","wb")
pickle.dump(betweenness_centrality_wg,f)
f.close()

f = open("../data/betweenness_centrality_sg.pkl","wb")
pickle.dump(betweenness_centrality_sg,f)
f.close()
