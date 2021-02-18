import pandas as pd
import numpy as np
import networkx as nx
import pickle

# read the file to create a dictionary with author key and paper list as value
f = open("../data/author_papers.txt","r")
papers_set = set()
author_paper = {}
for l in f:
    auth_paps = [int(paper_id.strip()) for paper_id in l.split(":")[1].replace("[","").replace("]","").replace("\n","").replace("\'","").replace("\"","").split(",")]
    author_paper[int(l.split(":")[0])] = auth_paps
f.close()

def get_n_papers(authors):
	n_papers = {}
	for node in nodes:
	    if node in author_paper:
	        n_papers[node] =  len(author_paper[node])
	    else:
	        n_papers = 0
	return n_papers

def get_average_coauthors_n_papers(nodes):
	average_coauthors_n_papers = {}
	for node in nodes:
	    neighbors = list(G.neighbors(node))
	    if len(neighbors)==0:
	        average_coauthors_n_papers[node] = 0
	    s = 0
	    for n in neighbors:
	        s += len(author_paper[n])
	    average_coauthors_n_papers[node] = s / len(neighbors)
	return average_coauthors_n_papers

# load graph
G = nx.read_edgelist('../data/collaboration_network.edgelist', delimiter=' ', nodetype=int)
nodes = list(G.nodes())

# compute the average number of papers and the average coauthors' number of papers for each author
n_papers = get_n_papers(nodes)
average_coauthors_n_papers = get_average_coauthors_n_papers(nodes)

f = open("../data/n_papers.pkl","wb")
pickle.dump(n_papers,f)
f.close()

f = open("../data/average_coauthors_n_papers.pkl","wb")
pickle.dump(average_coauthors_n_papers,f)
f.close()



