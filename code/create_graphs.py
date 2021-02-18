import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
import scipy as sp

# read the file to create a dictionary with author key and paper list as value
f = open("../data/author_papers.txt","r")
papers_set = set()
author_paper = {}
for l in f:
    auth_paps = [int(paper_id.strip()) for paper_id in l.split(":")[1].replace("[","").replace("]","").replace("\n","").replace("\'","").replace("\"","").split(",")]
    author_paper[int(l.split(":")[0])] = auth_paps
f.close()

# extract information about authors and papers
paper_author = {}
for k,v in author_paper.items():
    for x in v:
        paper_author.setdefault(int(x),[]).append(int(k))

G = nx.read_edgelist('../data/collaboration_network.edgelist', delimiter=' ', nodetype=int)

nodes = {k: v for v, k in enumerate(list(G.nodes()))}
inv_nodes = {v: k for k, v in nodes.items()}

authors = [int(a) for a in author_paper.keys()]
papers = [int(p) for ps in author_paper.values() for p in ps]

# Weighted graph
# create weighted graph
adj = nx.adjacency_matrix(G)
wadj = 0.5*adj.tolil()
for a in authors:
    ps = author_paper[a]
    for p in ps:
        auts = paper_author[p]
        for aut in auts:
            if a != aut:
                na = nodes[a]
                naut = nodes[aut]
                wadj[na,naut] += 1

WG = nx.from_scipy_sparse_matrix(wadj)
WG = nx.relabel_nodes(WG, inv_nodes)
nx.write_edgelist(WG, '../data/weighted_collaboration_network.edgelist', data=["weight"])


# Similarity graph
# read embeddings of abstracts
text_embeddings = pd.read_csv("../data/author_embedding_64.csv", header=None)
text_embeddings = text_embeddings.rename(columns={0: "authorID"})

# create dictionnary of embeddings
embs = {}
for author in authors:
    embs[author] = text_embeddings[text_embeddings["authorID"] == author]

# create similarity graph
n = G.number_of_nodes()
simadj = sp.sparse.csr_matrix((n,n)).tolil()
for author1 in authors:
    papers = author_paper[author1]
    for paper in papers:
        coauthors = paper_author[paper]
        for author2 in coauthors:
            if author1 != author2:
                emb1 = embs[author1]
                emb2 = embs[author2]
                n_author1 = nodes[author1]
                n_author2 = nodes[author2]
                val = cosine(emb1,emb2)
                simadj[n_author1,n_author2] = val

SG = nx.from_scipy_sparse_matrix(simadj)
SG = nx.relabel_nodes(SG, inv_nodes)
nx.write_multiline_adjlist(SG, '../data/sim_collaboration_network.adjlist')
