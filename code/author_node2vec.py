import os
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import ast
import re

node2vec_input = "../data/node2vec_wpg.nodevectors"
node2vec_output = "../data/author_node2vec_wpg.csv"

# read the file to create a dictionary with author key and paper list as value
f = open("../data/author_papers.txt","r")
papers_set = set()
d = {}
for l in f:
    auth_paps = [paper_id.strip() for paper_id in l.split(":")[1].replace("[","").replace("]","").replace("\n","").replace("\'","").replace("\"","").split(",")]
    d[l.split(":")[0]] = auth_paps
f.close()

# read the node2vec embeddings of each paper
papers = KeyedVectors.load_word2vec_format(node2vec_input)

def most_popular_paper(author):
    most_popular = np.zeros(64)
    best_popularity = 0
    for paper in d[author]:
        popularity = 0
        indicatrice = 0
        for co_author in G.neighbors(int(author)):
            if paper in d[str(co_author)]:
                popularity +=1
        if popularity > best_popularity:
            try:
                most_popular = papers[paper]
            except:
                indicatrice = 1
                continue
            best_popularity = indicatrice*best_popularity + (1-indicatrice)*popularity

    return most_popular
        
def weighted_mean(author):
    epsilon = 1e-10
    output = np.zeros(64)
    sum_of_popularity = 0
    for paper in d[author]:
        popularity = 0
        indicatrice = 1
        for co_author in G.neighbors(int(author)):
            if paper in d[str(co_author)]:
                popularity +=1
        try:
            output = papers[paper] * popularity
        except:
            indicatrice = 0
            continue
        sum_of_popularity += indicatrice*popularity
    output = output / (sum_of_popularity + epsilon)
    return output

#choose your node2vec embedding technique:
sum_ = True
weighted_mean = False
mean = False
most_popular = False

# To choose between NAN or vector of zero:
NAN = True


# the author node2vec embeddings is set to be the sum of its papers' node2vec embeddings
pattern = re.compile(r'(,){2,}')
df = open(node2vec_output,"w")
for author in d:
    
    # if you wish to have that the authors'vector corresponds to the vector of its most popular paper
    if most_popular:
        v = most_popular_paper(author)
    
    
    # if you wish to have that the authors'vector corresponds to the vector of its weighted mean
    if weighted_mean:
        v = weighted_mean(author)
        
        
    if sum_:
        v = np.zeros(64)
        for paper in d[author]:
            try:
                v += papers[str(paper)]
            except:
                continue
     
    
    if mean:
        v = np.zeros(64)
        c = 0
        for paper in d[author]:
            try:
                v+=papers[paper]
                c+=1
            except:
                continue
        if(c==0):
            c=1
       
    if NAN:
        if not v.any():
            v = np.nan * np.ones(64)
    
    df.write(author+","+",".join(map(lambda x:"{:.16f}".format(x), v))+"\n")
    
df.close()
