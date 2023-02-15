import os
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from graph_tool import Graph, centrality, collection, generation, topology
from graph_tool import stats as gt_stats
from scipy import optimize, stats
from tqdm import tqdm
from iminuit import Minuit
from iminuit.cost import LeastSquares
from iminuit.util import describe
#from numba import njit
import itertools
import graph_tool.all as gt
import networkx as nx
from networkx.algorithms import bipartite
from multiprocessing import Pool
from uncertainties import ufloat, umath
import pickle as pickle

# Function to find closeness and degree for graph_tool graph
# remove values if degree 0

def GetKC(g):
    """Get closeness and degree for graph_tool graph
    Parameters  
    ----------                  
    g : graph_tool graph
        Graph to be analyzed  
    Returns
    -------     
    k : array
        Degree of each node
    c : array
        Closeness of each node
    inv_c : array
        Inverse Closeness of each node
    mean_k : float
        Mean degree"""
    c = centrality.closeness(g).get_array()
    k = g.get_total_degrees(g.get_vertices())
    # Get mean degree
    mean_k = np.mean(k)
    # remove values if degree 0
    c = c[k > 0]
    k = k[k > 0]
    # remove values if closeness is nan
    k = k[~np.isnan(c)]
    c = c[~np.isnan(c)]
    # Get inverse closeness
    inv_c = 1/c
    return k, c, inv_c, mean_k

def GetK_2C(g):
    """Get closeness and degree for graph_tool graph
    Parameters  
    ----------                  
    g : graph_tool graph
        Graph to be analyzed  
    Returns
    -------     
    k : array
        Degree of each node
    c : array
        Closeness of each node
    inv_c : array
        Inverse Closeness of each node
    mean_k : float
        Mean degree"""
    c = centrality.closeness(g).get_array()
    k_1, k_2, vs = get_all_second_degree(g)
    k_2 = np.asarray(k_2)
    # Get mean degree
    mean_k_2 = np.mean(k_2)
    # remove values if degree 0
    c = c[k_2 > 0]
    k_2 = k_2[k_2 > 0]
    # remove values if closeness is nan
    k_2 = k_2[~np.isnan(c)]
    c = c[~np.isnan(c)]
    # Get inverse closeness
    inv_c = 1/c
    return k_2, c, inv_c, mean_k_2

# Function to get K and inverse c for bipartite graph
# Need to split into two groups
def GetKC_bipartite(g, to_print=False):
    """Get closeness and degree for bipartite graph_tool graph
    Parameters  
    ----------                  
    g : graph_tool graph
        Graph to be analyzed  
    Returns
    -------     
    k_1 : array
        Degree of each node in group 1
    c_1 : array
        Closeness of each node in group 1
    inv_c_1 : array
        Inverse Closeness of each node in group 1
    k_2 : array
        Degree of each node in group 2
    c_2 : array
        Closeness of each node in group 2
    inv_c_2 : array
        Inverse Closeness of each node in group 2
    mean_k_1 : float
        Mean degree of group 1
    mean_k_2 : float
        Mean degree of group 2"""
    k, c, inv_c, mean_k = GetKC(g)
    # get the bipartite sets
    test, partition = topology.is_bipartite(g, partition=True)

    # split into groups
    partition_array = partition.get_array()
    # find number of 1s and 0s
    num_1s = 0
    num_0s = 0
    for i in partition_array:
        if i == 1:
            num_1s += 1
        else:
            num_0s += 1
    
    if to_print:
        print(test)
        #print(partition.get_array())
        print('breakdown: ',num_1s, num_0s)

    # split into two groups
    k_1 = k[partition_array == 0]
    c_1 = c[partition_array == 0]
    inv_c_1 = inv_c[partition_array == 0]
    # Get mean of second 
    mean_k_1 = np.mean(k_1)
     
    # split into two groups
    k_2 = k[partition_array == 1]
    c_2 = c[partition_array == 1]
    inv_c_2 = inv_c[partition_array == 1]
    # Get mean of second
    mean_k_2 = np.mean(k_2)
    return k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2


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
    #print(f"Time taken: {end - start}")
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
    
    end = time.perf_counter()

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

# function to aggregate over x to find mean and standard error in y
#           --> remove if only appears once
def aggregate_dict(x, y):
    """Aggregate over x to find mean and standard deviation in y
    Parameters  
    ----------                  
    x : array
        x data
    y : array
        y data
    Returns
    -------     
    res : dictionary
        Dictionary containing x, y mean, y standard error, y std and y counts"""
    # x is k for us, y is inv_c.
    # we want to aggregate over k to find mean and standard error in inv_c
    # for each value of k. This is to allow goo
    x_mean, counts = np.unique(x,return_counts=True)
    # Prepare empty arrays
    y_mean = np.zeros(len(x_mean))
    y_err = np.zeros(len(x_mean))
    y_std = np.zeros(len(x_mean))
    y_counts = np.zeros(len(x_mean))
    # Loop over unique values of x
    # and find mean and standard error in y for each x
    for i in range(len(x_mean)):
        y_mean[i] = np.mean(y[x == x_mean[i]])
        y_std[i] = np.std(y[x == x_mean[i]])
        y_err[i] = y_std[i]/np.sqrt(counts[i])
        y_counts[i] = counts[i]
    res = {x_mean[i]:[y_mean[i],y_err[i],y_std[i],y_counts[i]] for i in range(len(x_mean))}
    return res


# Function to perform pearson test on unaggregated data.
def pearson(x, y):
    """Perform pearson test on unaggregated data.
    Parameters  
    ----------                  
    x : array
        x data
    y : array
        y data
    Returns
    -------     
    r : float
        Pearson correlation coefficient
    p : float
        p-value"""
    # quick scipy pearson
    r, p = stats.pearsonr(x, y)
    return r, p

# Function to perform rank correlation test on unaggregated data.
def spearman(x, y):
    """Perform rank correlation test on unaggregated data.
    Parameters  
    ----------                  
    x : array
        x data
    y : array
        y data
    Returns
    -------     
    r : float
        Spearman correlation coefficient
    p : float
        p-value"""
    # quick scipy spearman
    r, p = stats.spearmanr(x, y)
    return r, p

# Function to fit to for unipartite 1st order
# and to perform straight line fit to unaggregated data
def Tim(k, a, b):
    return -a*np.log(k) + b

# Function(s) describing model for bipartite graphs
def Harry_1(k, a, b, alpha):
    return -2*np.log(k*(1+np.exp(b)))/(a+b)+alpha

def Harry_2(k, a, b, alpha):
    return -2*np.log(k*(1+np.exp(a)))/(a+b)+alpha

def Tim_2(k_2, a, b):
    return -a*np.log(k_2) + b

def HO_Tim(k,a,b,N):
    pho = 1 - k/(N-1)
    return (-a*np.log(k) + b)*pho + (1-pho)*(1-a)

# Function Descibing analytic relation with second degree
#
#
#---------TO BE ADDED----------------
#
#

# Function to perform fit to specified function using curve_fit
def fitter(k,inv_c,function,N = None, to_print=False):
    """Perform fit to specified function using scipy curve_fit
    Parameters
    ----------
    k : array
        Degree of each node
    inv_c : array   
        Inverse Closeness of each node
    function : function
        Function to fit to
    to_print : bool
        Print results
    Returns
    ------- 
    popt[0] : float         
        a               
    popt[1] : float
        b
    pcov[0,0] : float
        a error
    pcov[1,1] : float
        b error
    """
    if N == None:
        popt, pcov = optimize.curve_fit(function, k, inv_c)
    else:
        function_2 = lambda k, a, b : function(k, a, b, N) 
        popt, pcov = optimize.curve_fit(function_2, k, inv_c)
    if to_print:
        print("gradient fit :", popt[0],"+/-",np.sqrt(pcov[0][0]))
        print("constant fit :",popt[1],"+/-",np.sqrt(pcov[1][1]))
    return popt, pcov #popt[0], np.sqrt(pcov[0][0]), popt[1], np.sqrt(pcov[1][1])

# Function to perform fit to bipartite analytic function, for the simultaneuos fit
# to both groups. Can do with both iminuit and scipy curve fit 

def fitter_test_dual(k1,k2,inv_c1,inv_c2,funcA,funcB,to_print=False):
    """Perform fit to bipartite analytic function, for the simultaneuos fit
    to both groups. Can do with both iminuit and scipy curve fit, this one is in iminuit
    Parameters
    ----------
    k1 : array
        Degree of each node in group 1
    k2 : array
        Degree of each node in group 2
    inv_c1 : array
        Inverse Closeness of each node in group 1
    inv_c2 : array
        Inverse Closeness of each node in group 2
    funcA : function
        Function to fit to group 1
    funcB : function
        Function to fit to group 2
    to_print : bool
        Print results
    Returns
    -------
    popt : array
        a, b, alpha
    errs : array
        a error, b error, alpha error
    """
    # Have to define errors. This is janky, but it works temporarily
    err_est_1 = 0.01#np.sqrt(inv_c1)
    err_est_2 = 0.01#np.sqrt(inv_c2)
    combined_LS = LeastSquares(k1,inv_c1,err_est_1, funcA) + LeastSquares(k2,inv_c2,err_est_2, funcB)
    print(f"{describe(combined_LS)=}")
    m = Minuit(combined_LS, a=1, b=1,alpha=1)
    m.migrad()
    if to_print:
        print(m.values)
    popt = [m.values['a'], m.values['b'], m.values['alpha']]
    errs = [m.errors['a'], m.errors['b'], m.errors['alpha']]
    return popt, errs

def fitter_test_dual_curve_fit(k1,k2,inv_c1,inv_c2,funcA,funcB,to_print=False):
    """Perform fit to bipartite analytic function, for the simultaneuos fit
    to both groups. Can do with both iminuit and scipy curve fit, this one is in iminuit
    Parameters
    ----------
    k1 : array
        Degree of each node in group 1
    k2 : array
        Degree of each node in group 2
    inv_c1 : array
        Inverse Closeness of each node in group 1
    inv_c2 : array
        Inverse Closeness of each node in group 2
    funcA : function
        Function to fit to group 1
    funcB : function
        Function to fit to group 2
    to_print : bool
        Print results
    Returns
    -------
    popt : array
        a, b, alpha
    errs : array
        a error, b error, alpha error
    """
    k = np.append(k1,k2)
    num_k1 = len(k1)
    num_k2 = len(k2)

    inv_c = np.append(inv_c1,inv_c2)

    def combo_func(k,a,b,alpha):
        data1 = funcA(k[:num_k1],a,b,alpha)
        data2 = funcB(k[num_k1:],a,b,alpha)
        return np.append(data1,data2)
    initial_guess = [1,1,1]

    popt, pcov = optimize.curve_fit(combo_func, k, inv_c, p0=initial_guess)
    if to_print:
        print(popt)

    errs = np.sqrt(np.diag(pcov))
    return popt, errs


# Function to perform chi square test on unaggregated data using standard deviation
def red_chi_square(k, inv_c, function, popt, stats_dict):
    """Perform chi square test on unaggregated data
    Parameters  
    ----------                  
    k : array
        Degree of each node
    inv_c : array
        Inverse Closeness of each node
    function : function
        Function to fit to
    popt : array
        Optimal parameters of fit
    sigma_dict : dict
        Dictionary of statistics (including standard deviation) 
        for given degree
    
    Returns
    -------     
    chi : float
        Chi square value
    p : float
        p-value"""
    #initialise new lists for when counts>1
    sigmas = []
    new_inv_c = []
    new_k = []
    for i in range(len(inv_c)):
        # Get sigma and count for given degree in array 
        sigma = stats_dict[k[i]][2]
        count = stats_dict[k[i]][3]
        # Need to not use if sigma = 0 (possible for identical nodes)
        if sigma > 0.01:
            # if count > 1 add to list as standard deviation is not 0
            if count>1:
                # add to new lists
                sigmas.append(sigma)
                new_inv_c.append(inv_c[i])
                new_k.append(k[i])
    # perform chi square test
    sigmas = np.asarray(sigmas)
    new_inv_c = np.asarray(new_inv_c)
    new_k = np.asarray(new_k)
    expected_inv_c = function(new_k, *popt)
    chi = np.sum(((new_inv_c - expected_inv_c)**2)/(sigmas**2))
    # get reduced chi square
    f = len(popt)
    rchi = chi/(len(new_k)-f)
    return rchi

# Function to get an analytical value for beta from z_fit

def beta_fit(a,a_err,N):
    """
    Parameters
    --------
    z : z_fit
        Fitted z from another function
    z_err : z_fit error
            z error to be propagated to give beta fit error
    N : Number of nodes of the graph being analysed
    Returns
    --------
    beta_fit : Fitted value for beta using analytical equation
    beta_fit_err : propagated beta fit error from z fit error
    """

    # Tims analytical equation for beta as a function of z_fit and N

    #x = ufloat(0.20, 0.01)

    a_1 = ufloat( a, a_err )
    z = umath.exp(1/a_1)
    z_err = z.s
    z_fit = np.exp(1/a)

    z_fit = ufloat( z_fit, z_err )
    beta_fit = umath.log(z_fit-1)/umath.log(z_fit) - 1/(z_fit-1) + umath.log(N)/umath.log(z_fit)

    beta_fit_err = beta_fit.s
    beta_fit = beta_fit.n #np.log(z_fit-1)/np.log(z_fit) - 1/(z_fit-1) + np.log(N)/np.log(z_fit)

    return beta_fit, beta_fit_err

def gamma_fit(z_fit,z_err,N):

    gamma_fit,gamma_fit_err = beta_fit(z_fit,z_err,N)

    return gamma_fit + 1, gamma_fit_err

# Function to do all above for given graph

def process(g,type,Real,Name=None):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    g : graph_tool graph
        Graph to be analyzed
    Returns
    -------     
    k : array
        Degree of each node
    c : array
        Closeness of each node
    a : float
        a
    a_err : float
        a error
    b : float
        b
    b_err : float
        b error
    rchi : float
        Reduced chi square value
    p : float
        p-value
    r : float
        Pearson correlation coefficient
    rp : float
        p-value
    rs : float
        Spearman correlation
    rsp : float
        p-value"""

    if type == 1:
        if Real:
            if os.path.exists('Data/GetKC/'+Name+'.pkl'):
                with open('Data/GetKC/'+Name+'.pkl','rb') as f:
                    k, c, inv_c, mean_k = pickle.load(f)
            else:
                os.makedirs('Data/GetKC',exist_ok=True)
                k, c, inv_c, mean_k = GetKC(g)
                with open('Data/GetKC/'+Name+'.pkl','wb') as f:
                    pickle.dump((k, c, inv_c, mean_k),f)
                
        else:
            k, c, inv_c, mean_k = GetKC(g)
        function = Tim
        popt, pcov = fitter(k, inv_c, function)
        statistics_dict = aggregate_dict(k, inv_c)
        rchi = red_chi_square(k, inv_c,function, popt,statistics_dict)
        r, rp = pearson(k, c)
        rs, rsp = spearman(k, c)
        return k, c, popt,pcov, rchi, r, rp, rs, rsp , statistics_dict, mean_k

    elif type == 3:
        if Real:
            if os.path.exists('Data/GetKC/'+Name+'.pkl'):
                with open('Data/GetKC/'+Name+'.pkl','rb') as f:
                    k, c, inv_c, mean_k = pickle.load(f)
            else:
                os.makedirs('Data/GetKC',exist_ok=True)
                k, c, inv_c, mean_k = GetKC(g)
                with open('Data/GetKC/'+Name+'.pkl','wb') as f:
                    pickle.dump((k, c, inv_c, mean_k),f)
                
        else:
            k, c, inv_c, mean_k = GetKC(g)
        function = HO_Tim
        N = len(g.get_vertices())
        popt, pcov = fitter(k, inv_c, function, N = N)
        popt = [popt[0],popt[1],N]
        statistics_dict = aggregate_dict(k, inv_c)
        rchi = red_chi_square(k, inv_c,function, popt,statistics_dict)
        r, rp = pearson(k, c)
        rs, rsp = spearman(k, c)
        return k, c, popt,pcov, rchi, r, rp, rs, rsp , statistics_dict, mean_k

    else:
        if Real:
            if os.path.exists('Data/GetK_2C/'+Name+'.pkl'):
                with open('Data/GetK_2C/'+Name+'.pkl','rb') as f:
                    k_2, c, inv_c, mean_k_2 = pickle.load(f)
            else:
                os.makedirs('Data/GetK_2C',exist_ok=True)
                k_2, c, inv_c, mean_k_2 = GetK_2C(g)
                with open('Data/GetK_2C/'+Name+'.pkl','wb') as f:
                    pickle.dump((k_2, c, inv_c, mean_k_2),f)
        else:
            k_2, c, inv_c, mean_k_2 = GetK_2C(g)
        function = Tim_2
        popt, pcov = fitter(k_2, inv_c, function)
        statistics_dict = aggregate_dict(k_2, inv_c)
        rchi = red_chi_square(k_2, inv_c,function, popt,statistics_dict)
        r, rp = pearson(k_2, c)
        rs, rsp = spearman(k_2, c)
        return k_2, c, popt,pcov, rchi, r, rp, rs, rsp , statistics_dict, mean_k_2

# Function to process Bipartite graphs
def process_bipartite(g,Real = False,Name=None):
    """Perform all analysis on bipartite graph
    Parameters
    ----------
    g : graph_tool graph
        Graph to be analyzed
    Returns
    -------
    output : list
        List of all outputs from processing
    """
    if Real:
        if os.path.exists('Data/GetKC_bipartite/'+Name+'.pkl'):
            with open('Data/GetKC_bipartite/'+Name+'.pkl','rb') as f:
                k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2 = pickle.load(f)
        else:
            os.makedirs('Data/GetKC_bipartite',exist_ok=True)
            k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2 = GetKC_bipartite(g)
            with open('Data/GetKC_bipartite/'+Name+'.pkl','wb') as f:
                pickle.dump((k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2),f)
    else:
        # Get degree and closeness for each node in two groups
        k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2 = GetKC_bipartite(g)
    
    # Fit to both groups - can use either scipy curve fit or iminuit
    popt, errs = fitter_test_dual_curve_fit(k_1,k_2,inv_c_1,inv_c_2, Harry_1, Harry_2)
    #popt1, errs1 = fitter_test_dual(k_1,k_2,inv_c_1,inv_c_2, Harry_1, Harry_2, to_print=to_print)
    a ,b ,alpha = popt

    # Get statistics for each degree
    statistics_dict_1 = aggregate_dict(k_1, inv_c_1)
    statistics_dict_2 = aggregate_dict(k_2, inv_c_2)

    # Perform chi square test on each group
    rchi_1 = red_chi_square(k_1, inv_c_1, Harry_1, popt,statistics_dict_1)
    rchi_2 = red_chi_square(k_2, inv_c_2, Harry_2, popt,statistics_dict_2)

    # Perform correlation tests
    r1, rp1 = pearson(k_1, c_1)
    r2, rp2 = pearson(k_2, c_2)

    rs1, rsp1 = spearman(k_1, c_1)
    rs2, rsp2 = spearman(k_2, c_2)

    # Return results
    output = [k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2, 
                rchi_1, rchi_2, r1, r2, rs1, rs2, rp1, rp2, rsp1, rsp2, 
                popt, errs, statistics_dict_1, statistics_dict_2]

    return output

# Function to generate BA model
# Here we specify an average degree to define the model
# by detemining m. 
def BA(n, av_deg):
    """Generate BA graph
    Parameters  
    ----------                  
    n : int
        Number of nodes
    av_deg : int
        Average degree
    Returns
    -------     
    g : graph_tool graph
        Generated graph"""
    # Determine 
    m = int(av_deg/2)
    g = generation.price_network(n, m, directed=False)
    return g


def ER(n, av_deg):
    """Generate ER graph
    Parameters  
    ----------                  
    n : int
        Number of nodes
    av_deg : float
        Average degree
    Returns
    -------     
    g : graph_tool graph
        Generated graph"""
    g = Graph()
    g.set_directed(False)
    p = av_deg/(n-1)
    for i in range(n):
        g.add_vertex()
    for a in range(n):
        for b in range(a+1,n):
            if (random.random() <p):
                g.add_edge(a,b)
    return g

# Function to generate Config-BA graph
# NEEDS TO BE ADDED
'''
def config_BA(n, av_deg):
    """Generate Config-BA graph
    Parameters  
    ----------                  
    n : int
        Number of nodes
    av_deg : int
        Average degree
    Returns
    -------     
    g : graph_tool graph
        Generated graph"""
    m = int(av_deg/2)
    g = generation.price_network(n, m, directed=False)
    g = generation.random_rewire(g, model="configuration")
    return g
'''

# Function to generate octopus/star graph
# NEEDS TO BE ADDED




#This makes an initial, complete bipartite graph ready to grow (preferentially)
#Every even node will be connected to every odd node.
#Probably best to choose a small n (initial number of nodes).
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
   
#Returns edge list (to put into graph tool) and the degree of each node in order
# I.e. from 0 to n-1  
# Now wrapping this in a function to make it load graph_tool graphs
def BipartiteBA(m_1,m_2,n, n_start=10):
    deg,edge_list=add(n_start,m_1,m_2,n)
   
    g = Graph()
    g.set_directed(False)
    g.add_edge_list(edge_list)
    return g


# Function to generate ER random bipartite graph
def BipartiteER(n1, n2, p):
    g = bipartite.random_graph(n1, n2,p, directed=False)

    nx.write_edgelist(g, "bipartite.txt", data=False)

    g = gt.load_graph_from_csv("bipartite.txt", directed=False, csv_options={"delimiter": " "})

    os.close("bipartite.txt")
    os.remove("bipartite.txt")
    
    return g


# Function to generate watts & strogatz bipartite graph
#
#
#----------TO BE ADDED----------
#
#
#

# Function to generate Config-BA bipartite graph
#
#
#----------TO BE ADDED----------
#
#
#
# Function to clean graph

def clean_graph(g, extra = True):
    """Clean graph
    Parameters  
    ----------                  
    g : graph_tool graph
        Graph to clean
    Returns
    -------     
    g : graph_tool graph
        Cleaned graph"""
    g = topology.extract_largest_component(g, directed=False, prune=True)
    if extra:
        gt_stats.remove_parallel_edges(g)
        gt_stats.remove_self_loops(g)
    return g


# Function to load graph from Netzschleuder
def load_graph(name, clean=True):
    """Load graph from pickle
    Parameters  
    ----------                  
    name : string
        Name of graph
    Returns
    -------     
    g : graph_tool graph
        Loaded graph"""
    g = collection.ns[name]
    if clean:
        g = topology.extract_largest_component(g, directed=False, prune=True)
        gt_stats.remove_parallel_edges(g)
        gt_stats.remove_self_loops(g)
    return g

def filter_num_verticies(names_df, num):
    """Filter graphs by number of verticies
    Parameters  
    ----------                  
    names_df : dataframe
        Dataframe of graph names and features
    num : int
        Number of verticies to filter by
    Returns
    -------     
    names_df : dataframe
        Filtered dataframe"""
    names_df = names_df.transpose()
    names_df = names_df.loc[names_df['num_vertices']<num,]
    names_df = names_df.transpose()
    return names_df

# Function to generate folders for saving data, plots
def MakeFolders(names, SubFolder):
    """Make folders for each network
    Parameters
    ----------
    names : list
        List of names of networks
    SubFolder : string
        Name of subfolder to create
    Returns
    -------
    None"""
    for i in names:
        path = 'Output/'+SubFolder +'/'+ i
        path = os.path.normpath(path)
        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)

# Function to unpack statistics dictionary
# Is a bit messy and inefficient, but works
# We use this in plotting - input is from aggregation_dict
def unpack_stat_dict(dict):
    """Unpack statistics dictionary
    Parameters
    ----------
    dict : dictionary
        Dictionary of statistics
    Returns
    -------
    ks_final : array
        Array of k values
    inv_c_mean : array
        Array of mean inverse closeness
    errs : array
        Array of errors on inverse closeness
    stds : array
        Array of standard deviations on inverse closeness
    counts : array
        Array of number of samples for each k"""

    ks = list(dict.keys())
    ks_final = []
    counts = []
    means = []
    stds = []
    errs = []
    for k in ks:
        vals = dict[k]
        if vals[3]>1:
            ks_final.append(k)
            means.append(vals[0])
            errs.append(vals[1])
            stds.append(vals[2])
            counts.append(vals[3])
    ks = np.asarray(ks)
    ks_final = np.asarray(ks_final)
    inv_c_mean = np.asarray(means)
    errs = np.asarray(errs)
    stds = np.asarray(stds)
    counts = np.asarray(counts)
    return ks_final, inv_c_mean, errs, stds, counts
        
# Function to write dataframes to html
def write_html(df, name, folder='Output'):
    """Write dataframe to html
    Parameters
    ----------
    df : dataframe
        Dataframe to be written
    name : string
        Name of file
    Returns
    -------
    None"""
    html = df.to_html()
    save_name = folder + '/' + name + '.html'
    text_file = open(save_name, "w")
    text_file.write(html)
    text_file.close()

# Function to write dataframes to latex
#
#
#----------TO BE ADDED----------
#
#
#