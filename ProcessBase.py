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

# Function to get K and inverse c for bipartite graph
# Need to split into two groups
def GetKC_bipartite(g):
    k, c, inv_c, mean_k = GetKC(g)
    # get the bipartite sets
    test, partition = topology.is_bipartite(g, partition=True)
    print(test)
    print(partition.get_array())
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
    
    print('breakdown: ',num_1s, num_0s)

    # split into two groups
    k_1 = k[partition_array == 0]
    c_1 = c[partition_array == 0]
    inv_c_1 = inv_c[partition_array == 0]
    mean_k_1 = np.mean(k_1)

    k_2 = k[partition_array == 1]
    c_2 = c[partition_array == 1]
    inv_c_2 = inv_c[partition_array == 1]
    mean_k_2 = np.mean(k_2)
    return k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2


# Function to get second degree
#
#
#--------TO BE ADDED----------------
#
#


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


# Function Descibing analytic relation with second degree
#
#
#---------TO BE ADDED----------------
#
#

# Function to perform fit to specified function using curve_fit
def fitter(k,inv_c,function,to_print=False):
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
    popt, pcov = optimize.curve_fit(function, k, inv_c)
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
        if sigma > 0.001:
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

# Function to do all above for given graph

def process(g,to_print=False):
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
    k, c, inv_c, mean_k = GetKC(g)
    function = Tim
    popt, pcov = fitter(k, inv_c, function, to_print=to_print)
    statistics_dict = aggregate_dict(k, inv_c)
    rchi = red_chi_square(k, inv_c,function, popt,statistics_dict)
    r, rp = pearson(k, c)
    rs, rsp = spearman(k, c)
    if to_print:
        print("Reduced chi square:", rchi)
        print("Pearson correlation:", r)
        print("Pearson p-value:", rp)
        print("Spearman correlation:", rs)
        print("Spearman p-value:", rsp)
    return k, c, popt,pcov, rchi, r, rp, rs, rsp , statistics_dict, mean_k

# Function to process Bipartite graphs
def process_bipartite(g,to_print=False):
    # Get degree and closeness for each node in two groups
    k_1, c_1, inv_c_1, k_2, c_2, inv_c_2, mean_k_1, mean_k_2 = GetKC_bipartite(g)
    
    # Fit to both groups - can use either scipy curve fit or iminuit
    popt, errs = fitter_test_dual_curve_fit(k_1,k_2,inv_c_1,inv_c_2, Harry_1, Harry_2, to_print=to_print)
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

    # Print results
    if to_print:
        print("Group 1 Reduced chi square:", rchi_1)
        print("Group 1 Pearson correlation:", r1)
        print("Group 1 Pearson p-value:", rp1)
        print("Group 1 Spearman correlation:", rs1)
        print("Group 1 Spearman p-value:", rsp1)
        print("Group 2 Reduced chi square:", rchi_2)
        print("Group 2 Pearson correlation:", r2)
        print("Group 2 Pearson p-value:", rp2)
        print("Group 2 Spearman correlation:", rs2)
        print("Group 2 Spearman p-value:", rsp2)
        print("a:", a)
        print("a error:", errs[0])
        print("b:", b)
        print("b error:", errs[1])
        print("alpha:", alpha)
        print("alpha error:", errs[2])

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
# Function to generate BA Bipartite graph
#
#
#----------TO BE ADDED----------
#
#
#

# Function to generate ER random bipartite graph
#
#
#----------TO BE ADDED----------
#
#
#


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

