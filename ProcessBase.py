from graph_tool import Graph, generation, centrality, collection
import time
import numpy as np
import pandas as pd
import scipy as sp 
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os
from pathlib import Path

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
        Inverse Closeness of each node"""
    c = centrality.closeness(g).get_array()
    k = g.get_total_degrees(g.get_vertices())
    #print(np.mean(k))
    c = c[k > 0]
    k = k[k > 0]
    k = k[~np.isnan(c)]
    c = c[~np.isnan(c)]
    inv_c = 1/c
    return k, c, inv_c

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
        Dictionary containing x, y, and y error"""
    x_mean, counts = np.unique(x,return_counts=True)
    y_mean = np.zeros(len(x_mean))
    y_err = np.zeros(len(x_mean))
    y_std = np.zeros(len(x_mean))
    y_counts = np.zeros(len(x_mean))
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
    r, p = stats.spearmanr(x, y)
    return r, p

# Function to fit to for unipartite 1st order
# and to perform straight line fit to unaggregated data
def func(k, a, b):
    return -a*np.log(k) + b

# Function describing analytic relation for bipartite graphs
#
#
#---------TO BE ADDED----------------
#
#

# Function Descibing analytic relation with second degree
#
#
#---------TO BE ADDED----------------
#
#


# Function to perform fit to analytic function
def fitter(k,inv_c,function,to_print=False):
    """Perform fit to specified function
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
        print("1/ln(z) fit :", popt[0],"+/-",np.sqrt(pcov[0][0]))
        print("B fit :",popt[1],"+/-",np.sqrt(pcov[1][1]))
    return popt[0], np.sqrt(pcov[0][0]), popt[1], np.sqrt(pcov[1][1])

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
        sigma = stats_dict[k[i]][2]
        count = stats_dict[k[i]][3]
        if sigma>0.001 and count>1:
            sigmas.append(sigma)
            new_inv_c.append(inv_c[i])
            new_k.append(k[i])
    sigmas = np.asarray(sigmas)
    new_inv_c = np.asarray(new_inv_c)
    new_k = np.asarray(new_k)
    expected_inv_c = function(new_k, *popt)
    chi = np.sum(((new_inv_c - expected_inv_c)**2)/(sigmas**2))
    rchi = chi/(len(new_k)-2)
    return rchi

# Function to do all above for given graph

def process(g, function,to_print=False):
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
    k, c, inv_c = GetKC(g)
    a, a_err, b, b_err = fitter(k, inv_c, function, to_print=to_print)
    statistics_dict = aggregate_dict(k, inv_c)
    rchi = red_chi_square(k, inv_c, function, [a, b],statistics_dict)
    r, rp = pearson(k, c)
    rs, rsp = spearman(k, c)
    if to_print:
        print("Reduced chi square:", rchi)
        print("Pearson correlation:", r)
        print("Pearson p-value:", rp)
        print("Spearman correlation:", rs)
        print("Spearman p-value:", rsp)
    return k, c, a, a_err, b, b_err, rchi, r, rp, rs, rsp , statistics_dict

# Function to generate BA, ER, Config-BA

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
# Function to generate Bipaerite graph
#
#
#----------TO BE ADDED----------
#
#
#


# Function to load graph from Netzschleuder
def load_graph(name):
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