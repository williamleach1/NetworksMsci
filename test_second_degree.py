
import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import time
import ProcessBase as pb

class VisitorSecondDegree(gt.BFSVisitor):

    def __init__(self, pred, dist):

        self.pred = pred
        self.dist = dist

    def tree_edge(self, e):
        current_dist = self.dist[e.source()] + 1
        # If alread visted all second degree neighbours, stop
        if current_dist > 2:
            raise StopIteration
        else:
            self.pred[e.target()] = int(e.source())
            self.dist[e.target()] = current_dist

class VisitorAll(gt.BFSVisitor):
    def __init__(self, pred, dist):

        self.pred = pred
        self.dist = dist

    def tree_edge(self, e):
        current_dist = self.dist[e.source()] + 1
        # If alread visted all second degree neighbours, stop
        self.pred[e.target()] = int(e.source())
        self.dist[e.target()] = current_dist


# add vertices and edges to the graph
def get_second_degree(g, v):
    '''
    Get counts at second degree (and first degree) from a vertex
    '''
    # new property for distance from bob
    dist = g.new_vertex_property("int")
    # new property for predecessor - along with distance specifies the tree
    pred = g.new_vertex_property("int64_t")
    # complete a bfs_search from bob 
    # This is a bit of a hack, but it works to ensure only second degree, not full bfs
    try:
        gt.bfs_search(g, v, VisitorSecondDegree(pred, dist))
    except StopIteration:
        pass
    # Find counts for distance 1 and 2
    distances = []
    for u in g.vertices():
        distances.append(dist[u])
    distances = np.array(distances)
    first_degree = np.sum(distances == 1)
    second_degree = np.sum(distances == 2)
    return first_degree, second_degree, v


def get_counts_at_distances(g, v):
    ''''
    Get counts at each distance from a vertex
    '''
    # new property for distance from bob
    dist = g.new_vertex_property("int")
    # new property for predecessor - along with distance specifies the tree
    pred = g.new_vertex_property("int64_t")
    # complete a bfs_search from bob 
    gt.bfs_search(g, v, VisitorAll(pred, dist))
    # Find counts at each distance
    distances = []
    for u in g.vertices():
        distances.append(dist[u])
    distances = np.array(distances)
    distances = np.sort(distances)
    distances, counts = np.unique(distances, return_counts=True)
    return distances, counts, v

# Find second degree for all vertices
def get_all_second_degree(g):
    k1s = []
    k2s = []
    vs = []
    start = time.perf_counter()
    '''
    pbar = tqdm((range(g.num_vertices())))
    for v in pbar:
        k1, k2, v = get_second_degree(g, v)
        k1s.append(k1)
        k2s.append(k2)
        vs.append(v)
    '''
    with Pool() as pool:
        results = pool.starmap(get_second_degree, [(g, v) for v in range(g.num_vertices())])
    for r in results:
        k1s.append(r[0])
        k2s.append(r[1])
        vs.append(r[2])
    
    end = time.perf_counter()
    print(f"Time taken: {end - start}")
    
    return k1s, k2s, vs

# Function to get counts at each distance for all vertices
def get_all_counts_with_distance(g):
    # Find counts at each distance for all vertices in g1 and plot average count at each distance
    vs = []
    distances = []
    counts = []
    start = time.perf_counter()
    # Parallel version
    with Pool() as pool:
        results = pool.starmap(get_counts_at_distances, [(g, v) for v in range(g.num_vertices())])
    for r in results:
        distances.append(r[0])
        counts.append(r[1])
        vs.append(r[2])
    
    '''
    pbar = tqdm((range(g.num_vertices())))
    for v in pbar:
        d, c, v = get_counts_at_distances(g, v)
        distances.append(d)
        counts.append(c)
        vs.append(v)
    '''
    end = time.perf_counter()
    print(f"Time taken: {end - start}")
    flat_distances = np.concatenate(distances).ravel()
    unique_distances = np.unique(flat_distances)

    mean_counts = []
    std_counts = []
    for u in unique_distances:
        counts_at_distance = []
        for i in range(len(distances)):
            ds = distances[i]
            index = np.where(ds == u)[0].tolist()
            if len(index) > 0:
                for j in index:
                    counts_at_distance.append(counts[i][j])
        counts_at_distance = np.array(counts_at_distance)
        mean = np.mean(counts_at_distance)
        mean_counts.append(mean)
        std = np.std(counts_at_distance)
        std_counts.append(std)
    return unique_distances, mean_counts, std_counts

if __name__ == "__main__":

    # Load graph
    g1 = pb.BA(5000, 20)
    print("Graph loaded")
    #unique_distances, mean_counts, std_counts = get_all_counts_with_distance(g1)
    k1s, k2s, vs = get_all_second_degree(g1)
    plt.plot(k1s, k2s, "o")
    plt.xlabel("First degree")
    plt.ylabel("Second degree")
    plt.show()
    k ,c, inv_c, mean_k = pb.GetKC(g1)
    # Tests
    print(len(k)==len(k1s))
    print(all(k==k1s))




    '''
    print(mean_counts)

    plt.plot(unique_distances, mean_counts, "o")
    plt.errorbar(unique_distances, mean_counts, yerr=std_counts, fmt="o")
    plt.xlabel("Distance")
    plt.ylabel("Counts")
    plt.show()
    '''




    
















