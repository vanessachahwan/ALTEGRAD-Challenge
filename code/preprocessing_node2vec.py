import os
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import networkx as nx
import ast
import re

node2vec_input = "../data/node2vec_sg.nodevectors"
node2vec_output = "../data/author_node2vec_sg_nan.csv"

# read the file to create a list of authors
G = nx.read_edgelist('../data/collaboration_network.edgelist', delimiter=' ', nodetype=int)

authors = list(G.nodes())

# read the node2vec embeddings of each paper
embeddings = KeyedVectors.load_word2vec_format(node2vec_input)

# To choose between NAN or vector of zero:
NAN = True


# the author node2vec embeddings is set to be the sum of its papers' node2vec embeddings
pattern = re.compile(r'(,){2,}')
df = open(node2vec_output,"w")
for author in authors:
    
    try:
        v = embeddings[str(author)]
    except:
        v = np.zeros(64)
       
    if NAN:
        if not v.any():
            v = np.nan * np.ones(64)
    
    df.write(str(author)+","+",".join(map(lambda x:"{:.8f}".format(x), v))+"\n")
    
df.close()
