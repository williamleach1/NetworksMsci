import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations
import os
import time
from scipy import stats
from tqdm import tqdm
import scipy as sp
from igraph import Graph
import pandas as pd

params = {
'font.size' : 16,
'font.family' : 'lmodern',
'axes.labelsize':16,
'legend.fontsize': 14,
'xtick.labelsize': 14,
'ytick.labelsize': 14,
}

plt.rcParams.update(params)

m,N = 3, 4000


def BA_me(m_,N_):
    nodes = [x for x in range(1,m_+1)]
    edges = list(combinations(nodes,2))
    t = 0 
    
    n = m_
    
    while len(nodes) < N_:
        new_node = n + 1
        nodes.append(new_node)
        not_allowed = [new_node]
        i=0
        while i < m_:
            edge_choice = random.choice(edges)
            node_choice = edge_choice[random.choice([0,1])]
            if node_choice in not_allowed:
                #print('Connection from ', new_node,' to ', node_choice,'\nnot allowed.')
                continue
            else:
                not_allowed.append(node_choice)
                edges.append((new_node,node_choice))
                i+=1
        n+=1
        t+=1
    #nodes, degrees = np.unique(edges,return_counts=True)
    return nodes, edges#, degrees

def BA_netx(m_,N_):
    G = nx.barabasi_albert_graph(N_,m_)
    return G

def BA_ig(m_,N_):
    return Graph.Barabasi(N_,m_)

start_me = time.time()
n,e = BA_me(m,N)
end_me = time.time()

start_netx = time.time()
G = BA_netx(m,N)
end_netx = time.time()

start_igraph = time.time()
G1 = BA_ig(m,N)
end_igraph = time.time()




print("Networkx Time = ",end_netx-start_netx)
print("my Code Time = ", end_me-start_me)
print("iGraph Time = ", end_igraph-start_igraph)
'''
start_close_netx = time.time()
#closeness = [close for (node,close) in kx.closeness_centrality(G)]
degrees = [val for (node, val) in G.degree()]
unique_degrees = np.unique(degrees)
inv_close = 1/np.asarray(list(nx.closeness_centrality(G).values()))
print(len(degrees),len(inv_close))
end_close_netx = time.time()
'''

start_close_igraph = time.time()
degrees2 = G1.degree()
inv_close2 = 1/np.asarray(G1.closeness())
print(len(degrees2),len(inv_close2))
end_close_igraph = time.time()

df = pd.DataFrame({"k":degrees2,"1/c":inv_close2})
df = df.groupby("k").agg({"1/c":['mean','std']})
df = df.xs('1/c', axis=1, drop_level=True)
df = df.reset_index('k')
df = df.rename(columns={"mean":"mean 1/c","std":"std 1/c"})

#print("Networkx closeness time: ",end_close_netx-start_close_netx)
print("iGraph closeness time: ",end_close_igraph-start_close_igraph)
'''
plt.plot(degrees,inv_close,'x')
plt.xscale('log')
plt.show()
plt.plot(degrees2,inv_close2,'x')
plt.xscale('log')
plt.show()
'''
ks = df.loc[:,"k"].tolist()
inv_c = df.loc[:,"mean 1/c"].tolist()
inv_c_err = df.loc[:,"std 1/c"].tolist()


plt.errorbar(ks,inv_c,yerr=inv_c_err,fmt='none',ecolor='black',capsize=2)
plt.plot(ks,inv_c,'ro')
plt.xscale("log")
plt.show()
