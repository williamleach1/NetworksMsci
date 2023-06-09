import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from copy import deepcopy

def bfs_layers(G, sources):
    """
    Generate the layers of a breadth-first-search starting at the given source

    This was taken from networkx as it was not workking in this version for some  reason
    """
    if sources in G:
        sources = [sources]

    current_layer = list(sources)
    visited = set(sources)

    for source in current_layer:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    while current_layer:
        yield current_layer
        next_layer = list()
        for node in current_layer:
            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    next_layer.append(child)
        current_layer = next_layer


number_per_ring = [1, 3, 10, 18, 8]
tree_edges_per_node_per_ring = [[3],[4,2,4],[1,2,2,2,3,1,2,1,3,1],[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,0]]

# create list of nodes in each ring
nodes_in_ring = []
for i in range(len(number_per_ring)):
    nodes_in_ring.append(list(range(sum(number_per_ring[:i]),sum(number_per_ring[:i+1]))))

print(nodes_in_ring)

# go through each ring and add edges
edges = []
num_rings = len(nodes_in_ring)

for i in range(num_rings):
    # if not last ring
    if i < num_rings - 1:
        # go through each node in ring
        temp_targets = nodes_in_ring[i+1]
        print(temp_targets)
        n = 0
        for j in range(len(nodes_in_ring[i])):
            source = nodes_in_ring[i][j]
            for k in range(tree_edges_per_node_per_ring[i][j]):
                target = temp_targets[n]
                print((source,target))
                edges.append((source,target))
                n += 1
            

nodes = [i for i in range(sum(number_per_ring))]
print(nodes)
print(edges)
tree_edges = deepcopy(edges)


fig, ax = plt.subplots(1,1, figsize = (10,10))
#g = nx.watts_strogatz_graph(40,4,0.5)  #nx.karate_club_graph()



g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
layers = dict(enumerate(bfs_layers(g, [0])))




all_cols = ['lightcoral', 'cornflowerblue']
node_colours = []
node_label = {}
pos_bipartite = {}
#find layer of node and assign colour
# Even layers are red, odd layers are blue
A_counter = 0
B_counter = 0


for i in range(len(nodes)):
    for j in range(len(layers)):
        if nodes[i] in layers[j]:
            if j % 2 == 0:
                node_colours.append(all_cols[0])
                pos_bipartite[nodes[i]] = np.array([0, A_counter])
                A_counter += 1
            else:
                node_colours.append(all_cols[1])
                pos_bipartite[nodes[i]] = np.array([10, B_counter])
                B_counter += 1

# add some random edges to the graph
# node in layer can only connect to nodes in layer above or below
# To do this, we need to find the layer of each node
# then we can add edges between nodes in the same layer and nodes in the layer above or below
# add edges between nodes in the same layer and nodes in the layer above or below
for i in range(len(layers)):
    # if not first layer
    if i > 0 and i < len(layers) - 1:
        # go through each node in layer
        for j in range(len(layers[i])):
            source = layers[i][j]
            # targets are list of nodes in layer above or below
            targets = layers[i-1] + layers[i+1]
            # randomly select number of edges to add
            num_edges = np.random.randint(0, 2)
            # randomly select targets
            targets = np.random.choice(targets, num_edges, replace = False)
            # add edges
            for k in range(len(targets)):
                g.add_edge(source, targets[k])
    if i == len(layers):
        for j in range(len(layers[i])):
            source = layers[i][j]
            # flatten list of nodes in layer below, and same layer
            targets = layers[i-1] + layers[i]
            # remove source node from list
            targets.remove(source)
            # randomly select number of edges to add
            num_edges = np.random.randint(0, 3)
            # randomly select targets
            targets = np.random.choice(targets, num_edges, replace = False)
            # add edges
            for k in range(len(targets)):
                g.add_edge(source, targets[k])


edges = g.edges()
sorted_tree_edges = [tuple(sorted(e)) for e in tree_edges]
widths, cols, alphas = [], [], []
for e in edges:
    sort_e = tuple(sorted(e))
    if e in sorted_tree_edges:
        widths.append(2)
        cols.append('k')
        alphas.append(1)
    else:
        widths.append(0.25)
        cols.append('k')
        alphas.append(0.01)

# function to generate set number of equally spaced points on a circle of radius r
def circle(r, n):
    return [np.array([r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n)]) for i in range(n)]

# function to generate set number of equally spaced points on a circle of radius r
# then rotate them by same random angle
def circle_rand(r, n):
    c = circle(r, n)
    theta = np.random.uniform(0, 2 * np.pi)
    return [np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), c[i]) for i in range(n)]


pos = {}
for i in range(len(layers)):
    r = i
    n = len(layers[i])
    c = circle(r, n)
    for j in range(len(layers[i])):
        pos[layers[i][j]] = c[j]
        

# draw circles of radius 1, 2, 3, 4, 5
for i in range(len(layers)):
    c = plt.Circle((0, 0), i, linestyle = '--',color='k', fill=False, clip_on=False)
    ax.add_artist(c)


nx.draw(g, pos, ax = ax,node_color=node_colours, width = widths, edge_color = cols, linewidths = 1)

# add a custom legend with the node colours
groups = ['A', 'B']

for i in range(2):
    ax.scatter([], [], c=all_cols[i],s=200, label='Group ' + groups[i])
ax.legend(scatterpoints=1, frameon=False, labelspacing=1,loc='upper center', bbox_to_anchor=(0.5, 1.1),
            ncol=3, fancybox=True, shadow=True, fontsize = 18)

plt.savefig('NetworkDemoPlots/BipartiteRingPlot.png', dpi = 1200, bbox_inches = 'tight')
plt.show()

'''
listy = [1,2,3,4,5,6,7,8,9,10]

# make a bar plot of listy
plt.figure()
plt.bar(range(len(listy)), listy)
plt.show()

# test if g is bipartite
print(nx.is_bipartite(g))



# Also plot g with bipartite layout
fig, ax = plt.subplots(1,1, figsize = (10,10))
# custome pos dict for bipartite layout
# nodes in group A are at y = 0, nodes in group B are at y = 10 using pos_bipartite
# Randomly take nodes in group A and B and place them at x = 0 and x = 10 respectively

all_cols = ['lightcoral', 'cornflowerblue']

pos_bipartite = {}
A_counter, B_counter = 0, 0
for i in range(len(nodes)):
    if node_colours[i] == all_cols[0]:
        pos_bipartite[nodes[i]] = np.array([0, A_counter])
        A_counter += 1
    else:
        pos_bipartite[nodes[i]] = np.array([10, B_counter])
        B_counter += 1




nx.draw(g, pos_bipartite, ax = ax,node_color=node_colours, width = widths, edge_color = cols, linewidths = 1)


# add a custom legend with the node colours
groups = ['A', 'B']

for i in range(2):
    ax.scatter([], [], c=all_cols[i],s=200, label='Group ' + groups[i])
ax.legend(scatterpoints=1, frameon=False, labelspacing=1,loc='upper center', bbox_to_anchor=(0.5, 1.1),
            ncol=3, fancybox=True, shadow=True, fontsize = 18)

plt.savefig('NetworkDemoPlots/BipartiteRingPlotBipartite.png', dpi = 1200, bbox_inches = 'tight')
plt.show()


'''
# Generate a simple bipartite graph

all_cols = ['mediumseagreen', 'violet']

nodes_a = [1,2,3]

nodes_b = [6,7,8,9,10,11,12]

pos_bipartite = {}

node_colours = []
for i in range(len(nodes_a)):
    pos_bipartite[nodes_a[i]] = np.array([0,1+ i*2])
    node_colours.append('mediumseagreen')


for i in range(len(nodes_b)):
    pos_bipartite[nodes_b[i]] = np.array([10, i*1])
    node_colours.append('violet')



g = nx.Graph()

g.add_nodes_from(nodes_a, bipartite = 0)
g.add_nodes_from(nodes_b, bipartite = 1)

# add edges between nodes in different groups at random, contnue until all nodes are connected
while nx.is_connected(g) == False:
    # randomly select a node in group a
    node_a = np.random.choice(nodes_a)
    # randomly select a node in group b
    node_b = np.random.choice(nodes_b)
    # add edge between them
    g.add_edge(node_a, node_b)

# Plot the graph

fig, ax = plt.subplots(1,1, figsize = (10,10))

# custome pos dict for bipartite layout
# nodes in group A are at y = 0, nodes in group B are at y = 10 using pos_bipartite
# Randomly take nodes in group A and B and place them at x = 0 and x = 10 respectively


nx.draw(g, pos_bipartite, ax = ax,node_color=node_colours, linewidths = 1)

# add a custom legend with the node colours
groups = ['A', 'B']

for i in range(2):
    ax.scatter([], [], c=all_cols[i],s=200, label='Group ' + groups[i])
ax.legend(scatterpoints=1, frameon=False, labelspacing=1,loc='upper center', bbox_to_anchor=(0.5, 1.1),
            ncol=3, fancybox=True, shadow=True, fontsize = 18)

plt.savefig('NetworkDemoPlots/BipartiteRandomPlot.png', dpi = 1200, bbox_inches = 'tight')

plt.show()
