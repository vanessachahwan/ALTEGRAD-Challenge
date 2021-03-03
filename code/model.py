import pandas as pd
import numpy as np
import networkx as nx
import pickle
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from gensim.models import KeyedVectors


# read training data
df_train = pd.read_csv('../data/train.csv', dtype={'authorID': np.int64, 'h_index': np.float32})
n_train = df_train.shape[0]


# read test data
df_test = pd.read_csv('../data/test.csv', dtype={'authorID': np.int64})
n_test = df_test.shape[0]


# read collaboration graph
G = nx.read_edgelist('../data/collaboration_network.edgelist', delimiter=' ', nodetype=int)


# read weighted collaboration graph
WG = nx.read_edgelist("../data/weighted_collaboration_network.edgelist", nodetype=int, data=(("weight", float),))


# read author similarity graph
SG = nx.read_multiline_adjlist("../data/author_similarity_network.adjlist", nodetype=int)


nodes = {k: v for v, k in enumerate(list(G.nodes()))}


# compute graph features for each node
avg_neighbor_degree_wg = nx.average_neighbor_degree(WG)
avg_neighbor_degree_g = nx.average_neighbor_degree(G)
core_number_g = nx.core_number(G)
page_rank_g = nx.pagerank(G)
page_rank_wg = nx.pagerank(WG)
avg_neighbor_degree_sg = nx.average_neighbor_degree(SG)
page_rank_sg = nx.pagerank(SG)
eigenvector_centrality_sg = nx.eigenvector_centrality(SG)


# load precomputed features for each node
f = open("../data/n_papers.pkl", "rb")
n_papers = pickle.load(f)
f.close()

f = open("../data/average_coauthors_n_papers.pkl", "rb")
average_coauthors_n_papers = pickle.load(f)
f.close()

f = open("../data/betweenness_centrality_g.pkl", "rb")
betweenness_centrality_g = pickle.load(f)
f.close()

f = open("../data/betweenness_centrality_wg.pkl", "rb")
betweenness_centrality_wg = pickle.load(f)
f.close()

f = open("../data/clustering_wg.pkl", "rb")
clustering_wg = pickle.load(f)
f.close()

f = open("../data/clustering_sg.pkl", "rb")
clustering_sg = pickle.load(f)
f.close()


# load Node2Vec embeddings obtained from WG
node2vec_wg = KeyedVectors.load_word2vec_format('../data/node2vec_wg.nodevectors')


# load Node2Vec embeddings obtained from SG
n2v_sg = pd.read_csv("../data/author_node2vec_sg.csv", header=None)
n2v_sg = n2v_sg.rename(columns={0: "authorID"})


# load Node2Vec embeddings obtained from PG
n2v_pg = pd.read_csv("../data/author_node2vec_pg.csv", header=None)
n2v_pg = n2v_pg.rename(columns={0: "authorID"})


# load Node2Vec embeddings obtained from WPG
n2v_wpg = pd.read_csv("../data/author_node2vec_wpg.csv", header=None)
n2v_wpg = n2v_wpg.rename(columns={0: "authorID"})


# read embeddings of abstracts obtained with Doc2Vec PV-DM
text_embeddings_dm = pd.read_csv("../data/author_embedding_64_dm.csv", header=None)
text_embeddings_dm = text_embeddings_dm.rename(columns={0: "authorID"})


# read embeddings of abstractsobtained with Doc2Vec PV-DBOW
text_embeddings_dbow = pd.read_csv("../data/author_embedding_64_dbow.csv", header=None)
text_embeddings_dbow = text_embeddings_dbow.rename(columns={0: "authorID"})


# preprocessing: fill nan values by zeros
n2v_sg = n2v_sg.fillna(0)
n2v_pg = n2v_pg.fillna(0)
n2v_wpg = n2v_wpg.fillna(0)
text_embeddings_dm = text_embeddings_dm.fillna(0)
text_embeddings_dbow = text_embeddings_dbow.fillna(0)


n_dim = 64


# create the training matrix. each node is represented as a vector of features:
# (1-2-3) its degree, (4-5-6) the average degree of its neighbors, 
# (7) its core number, (8-9-10) its page rank, (11-12) its betweenness centrality, 
# (13-14) its clustering coefficient, (15) its eigenvector centrality,
# (16) the number of written papers (cited), (17) the average number of written papers of its neighbors/coauthors,
# (18-19-20-21) embeddings from Node2Vec, (22-23) text_embeddings from Doc2Vec
X_train = np.zeros((n_train, 17+6*n_dim))
y_train = np.zeros(n_train)
for i,row in df_train.iterrows():
    node = int(row['authorID'])
    index = nodes[node]
    X_train[i,0] = G.degree(node)
    X_train[i,1] = WG.degree(node)
    X_train[i,2] = SG.degree(node)
    X_train[i,3] = avg_neighbor_degree_g[node]
    X_train[i,4] = avg_neighbor_degree_wg[node]
    X_train[i,5] = avg_neighbor_degree_sg[node]
    X_train[i,6] = core_number_g[node]
    X_train[i,7] = page_rank_g[node]
    X_train[i,8] = page_rank_wg[node]
    X_train[i,9] = page_rank_sg[node]
    X_train[i,10] = betweenness_centrality_g[node]
    X_train[i,11] = betweenness_centrality_wg[node]
    X_train[i,12] = clustering_wg[node]
    X_train[i,13] = clustering_sg[node]
    X_train[i,14] = eigenvector_centrality_sg[node]
    X_train[i,15] = n_papers[node]
    X_train[i,16] = average_coauthors_n_papers[node]
    X_train[i,17:17+n_dim] = n2v_pg[n2v_pg["authorID"] == node].iloc[:,1:]
    X_train[i,17+n_dim:17+2*n_dim] = n2v_wpg[n2v_wpg["authorID"] == node].iloc[:,1:]
    X_train[i,17+2*n_dim:17+3*n_dim] = n2v_sg[n2v_sg["authorID"] == node].iloc[:,1:]
    X_train[i,17+3*n_dim:17+4*n_dim] = node2vec_wg[str(node)]
    X_train[i,17+4*n_dim:17+5*n_dim] = text_embeddings_dm[text_embeddings_dm["authorID"] == node].iloc[:,1:]
    X_train[i,17+5*n_dim:] = text_embeddings_dbow[text_embeddings_dbow["authorID"] == node].iloc[:,1:]
    y_train[i] = row['h_index']


# create the testing matrix. each node is represented as a vector of features:
# (1-2-3) its degree, (4-5-6) the average degree of its neighbors, 
# (7) its core number, (8-9-10) its page rank, (11-12) its betweenness centrality, 
# (13-14) its clustering coefficient, (15) its eigenvector centrality,
# (16) the number of written papers (cited), (17) the average number of written papers of its neighbors/coauthors,
# (18-19-20-21) embeddings from Node2Vec, (22-23) text_embeddings from Doc2Vec
X_test = np.zeros((n_test, 17+6*n_dim))
for i,row in df_test.iterrows():
    node = int(row['authorID'])
    index = nodes[node]
    X_test[i,0] = G.degree(node)
    X_test[i,1] = WG.degree(node)
    X_test[i,2] = SG.degree(node)
    X_test[i,3] = avg_neighbor_degree_g[node]
    X_test[i,4] = avg_neighbor_degree_wg[node]
    X_test[i,5] = avg_neighbor_degree_sg[node]
    X_test[i,6] = core_number_g[node]
    X_test[i,7] = page_rank_g[node]
    X_test[i,8] = page_rank_wg[node]
    X_test[i,9] = page_rank_sg[node]
    X_test[i,10] = betweenness_centrality_g[node]
    X_test[i,11] = betweenness_centrality_wg[node]
    X_test[i,12] = clustering_wg[node]
    X_test[i,13] = clustering_sg[node]
    X_test[i,14] = eigenvector_centrality_sg[node]
    X_test[i,15] = n_papers[node]
    X_test[i,16] = average_coauthors_n_papers[node]
    X_test[i,17:17+n_dim] = n2v_pg[n2v_pg["authorID"] == node].iloc[:,1:]
    X_test[i,17+n_dim:17+2*n_dim] = n2v_wpg[n2v_wpg["authorID"] == node].iloc[:,1:]
    X_test[i,17+2*n_dim:17+3*n_dim] = n2v_sg[n2v_sg["authorID"] == node].iloc[:,1:]
    X_test[i,17+3*n_dim:17+4*n_dim] = node2vec_wg[str(node)]
    X_test[i,17+4*n_dim:17+5*n_dim] = text_embeddings_dm[text_embeddings_dm["authorID"] == node].iloc[:,1:]
    X_test[i,17+5*n_dim:] = text_embeddings_dbow[text_embeddings_dbow["authorID"] == node].iloc[:,1:]


# regressor model
reg = LGBMRegressor(objective='mae',
                    boosting_type='dart',
                    colsample_bytree = 0.9365930147015064,
                    learning_rate = 0.08882564114326925,
                    max_depth = 10,
                    n_estimators = 100000,
                    num_leaves = 32,
                    reg_alpha = 0.7886706622516692,
                    reg_lambda = 0.510000982102748,
                    subsample = 0.5044671893327269, 
                    n_jobs=1)


# training the regressor and making a prediction
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)


# post-processing: take nearest integer value and bound it
y_pred = np.round(y_pred)

for i in range(len(X_test)):
    npapers = X_test[i, 15]
    if npapers < 10 and y_pred[i] > npapers:
        y_pred[i] = npapers
    if y_pred[i] <= 0:
        y_pred[i] = 1


# write the predictions to file
df_test['h_index_pred'].update(pd.Series(np.round_(y_pred, decimals=3)))
df_test.loc[:,["authorID","h_index_pred"]].to_csv('../data/test_predictions.csv', index=False)
