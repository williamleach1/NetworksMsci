import networkx as nx
from networkx.algorithms import bipartite
import graph_tool.all as gt
from graph_tool import topology, Graph
import os
import numpy as np
import itertools
import random

def ERBipartite(n, m, p):
    g = bipartite.random_graph(n, m,p, directed=False)

    nx.write_edgelist(g, "bipartite.txt", data=False)

    g = gt.load_graph_from_csv("bipartite.txt", directed=False, csv_options={"delimiter": " "})

    os.remove("bipartite.txt")
    
    return g

g = ERBipartite(1000, 100, 0.1)

test, part = topology.is_bipartite(g, partition=True)

print(test)

def bi(n):
    
    lis=[]
    for i in range(n):
        lis.append(i)

    edges = list(itertools.combinations(lis, 2))
    for e in edges:
        if (e[0]+e[1]) %2 == 0:
            edges.remove(e)
   
    deg=[]
    for i in range(n):
        deg_count=0
        for e in edges:
            if e[0]==i or e[1] == i:
                deg_count+=1
               
        deg.append(deg_count)
   
    weight=np.array(edges).flatten().tolist()
   
    return weight, n, edges


#BA preferential growth model for bipartite
#n_start is the size of the initialised complete bipartite graph
#m_1,_2 are the two different 'wedge' numbers for each type of node
#n is the number of nodes to be added preferentially
#@njit
def add(n_start,m_1,m_2,n):
   
    weighting,n_initial,edge_list=bi(n_start)
    #weighting=w
    number=n_initial-1
    #edge_list=edge
    for x in range(n):
        number+=1
        odd = [num for num in weighting if num % 2 != 0]
        even=[num for num in weighting if num % 2 == 0]
       
        if (number + 1) % 2 == 0:
            weight=even
            m=m_1
           
        else:
            weight = odd
            m=m_2
           
        for i in range(m):
            index=np.random.randint(0,len(weight))
            attach=weight[index]
            weighting.append(attach)
            weighting.append(number)
            weight.append(attach)
            edge_list.append((number,attach))
   
    deg=np.bincount(weighting)
   
    return deg,edge_list

def BipartiteBA(n_start,m_1,m_2,n):
    deg,edge_list=add(n_start,m_1,m_2,n)
   
    g = Graph()
    g.set_directed(False)
    g.add_edge_list(edge_list)
    return g


g2 = BipartiteBA(10, 5, 10, 1000)

test, part = topology.is_bipartite(g2, partition=True)

print('For Harry Bipartite:   ',test)


