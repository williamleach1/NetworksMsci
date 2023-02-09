
import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import time
import ProcessBase as pb

class VisitorSecondDegree(gt.BFSVisitor):
    '''
    We use a custom visitor that inherits from BFSVisitor
        and overrides the tree_edge method.
    This tree edge method is then called by the bfs_search.
    Needs new properties for distance and predecessor to be made for the graph
        but this is done in the funciton.
    Used graph_tool documentation to start.
    '''
    def __init__(self, pred, dist):

        self.pred = pred
        self.dist = dist

    def tree_edge(self, e):
        current_dist = self.dist[e.source()] + 1
        # If alread visted all second degree neighbours, stop
        # This makes the function faster for large graphs
        if current_dist > 2:
            raise StopIteration
        else:
            # If not add to the tree
            self.pred[e.target()] = int(e.source())
            self.dist[e.target()] = current_dist

class VisitorAll(gt.BFSVisitor):
    '''
    We use a custom visitor that inherits from BFSVisitor
        and overrides the tree_edge method.
    This tree edge method is then called by the bfs_search.
    Needs new properties for distance and predecessor to be made for the graph
        but this is done in the funciton.
    This is the full BFS version and does not stop until all nodes visted.
    Used graph_tool documentation to start.
    '''
    def __init__(self, pred, dist):
        # Initialise
        self.pred = pred
        self.dist = dist

    def tree_edge(self, e):
        current_dist = self.dist[e.source()] + 1
        # Add to the tree
        self.pred[e.target()] = int(e.source())
        self.dist[e.target()] = current_dist


# add vertices and edges to the graph
def get_second_degree(g, v):
    '''
    Get counts at second degree (and first degree) from a vertex
    Parameters
    ----------
    g : graph_tool graph
        Graph to search
    v : graph_tool vertex
        Vertex to search from
    Returns
    -------
    first_degree : int
        Number of first degree neighbours
    second_degree : int
        Number of second degree neighbours
    v : graph_tool vertex
        Vertex that was searched from
    '''
    # new property for distance from current vertex
    dist = g.new_vertex_property("int")
    # new property for predecessor - along with distance specifies the tree
    pred = g.new_vertex_property("int64_t")
    # Start a bfs_search from the vertex
    # This is a bit of a hack, but it works to ensure only second degree, not full bfs
    try:
        gt.bfs_search(g, v, VisitorSecondDegree(pred, dist))
    except StopIteration:
        pass
    # Find counts for distance 1 and 2
    distances = []
    # iterate over all vertices to find distance from root
    for u in g.vertices():
        distances.append(dist[u])
    distances = np.array(distances)
    # First degree is distance one, second degree is distance two
    # We need the sum of these
    first_degree = np.sum(distances == 1)
    second_degree = np.sum(distances == 2)
    return first_degree, second_degree, v #Return


def get_counts_at_distances(g, v):
    ''''
    Get counts at each distance from a vertex
    Parameters
    ----------
    g : graph_tool graph
        Graph to search
    v : graph_tool vertex
        Vertex to search from
    Returns
    -------
    distances : np.array
        Distances from the vertex
    counts : np.array
        Number of vertices at each distance
    v : graph_tool vertex
        Vertex that was searched from
    
    '''
    # new property for distance from vertex
    dist = g.new_vertex_property("int")
    # new property for predecessor - along with distance specifies the tree
    pred = g.new_vertex_property("int64_t")
    # complete a bfs_search from bob 
    gt.bfs_search(g, v, VisitorAll(pred, dist))
    # Find counts at each distance using property
    # Could also use predecessor property (may need to modify branches) to specify branches
    distances = []
    for u in g.vertices():
        distances.append(dist[u])
    distances = np.array(distances)
    distances = np.sort(distances)
    # find counts at each distance
    distances, counts = np.unique(distances, return_counts=True)
    return distances, counts, v

# Find second degree for all vertices
def get_all_second_degree(g):
    '''
    Get counts at second degree (and first degree) from all vertices
    Use multiprocessing to speed up.
    Parameters
    ----------
    g : graph_tool graph
        Graph to search
    Returns
    -------
    k1s : list
        Number of first degree neighbours
    k2s : list
        Number of second degree neighbours
    vs : list
        Vertices that were searched from
    '''
    # Initialise lists
    k1s = []
    k2s = []
    vs = []
    start = time.perf_counter()
    # Keeping for if multiprocessing does not work later
    '''
    pbar = tqdm((range(g.num_vertices())))
    for v in pbar:
        k1, k2, v = get_second_degree(g, v)
        k1s.append(k1)
        k2s.append(k2)
        vs.append(v)
    '''
    # Parallel version  - this is much faster (by the factor of the number of cores)
    # Each core is doing a different vertex at any time (approximately)
    with Pool() as pool:
        results = pool.starmap(get_second_degree, [(g, v) for v in range(g.num_vertices())])
    # Unpack results by appending to lists
    for r in results:
        k1s.append(r[0])
        k2s.append(r[1])
        vs.append(r[2])
    
    end = time.perf_counter()
    print(f"Time taken: {end - start}")
    # Return lists - return vs to check we calculate other quantities from same vertex when comparing
    return k1s, k2s, vs

# Function to get counts at each distance for all vertices
def get_all_counts_with_distance(g):
    '''
    Get counts at each distance from all vertices
    Use multiprocessing to speed up.
    Parameters
    ----------
    g : graph_tool graph
        Graph to search
    Returns
    -------
    all_distances : list of np.arrays
        Distances from the vertex for each vertex
        i.e. distances[i] is the distances from vertex i
    all_counts : list of np.arrays
        Number of vertices at each distance
        i.e. counts[i] is the counts at each distance from vertex i
    vs : list 
        Vertices that were searched from
    '''
    # Find counts at each distance for all vertices
    vs = []
    all_distances = []
    all_counts = []
    start = time.perf_counter()
    # Parallel version  - this is much faster (by the factor of the number of cores)
    # Each core is doing a different vertex at any time (approximately)
    with Pool() as pool:
        results = pool.starmap(get_counts_at_distances, [(g, v) for v in range(g.num_vertices())])
    # Unpack results by appending to lists
    for r in results:
        all_distances.append(r[0])
        all_counts.append(r[1])
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
    # Return lists - return vs to check we calculate other quantities from same vertex when comparing
    return all_distances, all_counts, vs

def get_mean_and_std_at_distances(all_distances, all_counts):
    '''
    Get mean and standard deviation of counts at each distance
    Parameters
    ----------
    all_distances : list of np.arrays
        Distances from the vertex for each vertex
        i.e. distances[i] is the distances from vertex i
    all_counts : list of np.arrays
        counts at each distance for each vertex
        i.e. counts[i] is the counts at each distance from vertex i
    Returns
    -------
    unique_distances : np.array
        Distances from the vertex
    mean_counts : np.array
        Mean counts at each distance
    std_counts : np.array
        Standard deviation of counts at each distance
    '''
    # Flatten distances list of arrays into one array
    flat_distances = np.concatenate(all_distances).ravel()
    # Find unique distances
    unique_distances = np.unique(flat_distances)
    # Find mean and standard deviation of counts at each distance
    mean_counts = []
    std_counts = []
    # Loop over unique distances
    for u in unique_distances:
        # Find counts at each distance
        counts_at_distance = []
        for i in range(len(all_distances)):
            # Find indices where distance is equal to u (unique distance)
            ds = all_distances[i]
            index = np.where(ds == u)[0].tolist()
            # If there are any counts at this distance, append to list
            if len(index) > 0:
                for j in index:
                    counts_at_distance.append(all_counts[i][j])
        # Find mean and standard deviation of counts at this distance
        counts_at_distance = np.array(counts_at_distance)
        mean = np.mean(counts_at_distance)
        mean_counts.append(mean)
        std = np.std(counts_at_distance)
        std_counts.append(std)
    return unique_distances, mean_counts, std_counts

if __name__ == "__main__":
    # This is just to test
    # Load graph
    g1 = pb.BA(1000, 5)
    print("Graph loaded")
    all_distances, all_counts, vs = get_all_counts_with_distance(g1)
    unique_distances, mean_counts, std_counts = get_mean_and_std_at_distances(all_distances, all_counts)
    k1s, k2s, vs = get_all_second_degree(g1)
    plt.plot(k1s, k2s, "o")
    plt.xlabel("First degree")
    plt.ylabel("Second degree")
    plt.show()
    k ,c, inv_c, mean_k = pb.GetKC(g1)
    # Tests
    #print(len(k)==len(k1s))
    #print(all(k==k1s))

    print(mean_counts)

    # Now plot results for mean and individual counts to test
    for i in range(len(all_distances)):
        plt.plot(all_distances[i], all_counts[i], "rx", alpha= 0.3)
    plt.errorbar(unique_distances, mean_counts, yerr=std_counts, fmt="bo")
    plt.plot(unique_distances, mean_counts, "bo")
    plt.xlabel("Distance from root node")
    plt.ylabel("Number of nodes on ring")
    plt.show()
    




    
















