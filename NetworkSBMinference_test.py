import os
import networkx as nx
# Run for BA, ER and Config. return as dataframe
import seaborn as sns
from matplotlib import cm

from Plotter import *
from ProcessBase import *

from datetime import datetime
import csv
from graph_tool import generation, centrality, inference, collection
import graph_tool.all as gt






def SBM(sizes = [], ps = []):
    """Generate a stochastic block model graph
    Parameters
    ----------
    sizes : array
        Array of sizes of each block
    ps : array
        Array of probabilities of edges between blocks
    Returns
    -------
    g : graph
        Graph object"""

    membership = []
    for i,v in enumerate(sizes):
        for j in range(v):
            membership.append(i)
    membership = np.array(membership)
    



    g = generation.generate_sbm(membership, ps)
    return g, membership


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
    #c = c[k > 0]
    #k = k[k > 0]
    # remove values if closeness is nan
    #k = k[~np.isnan(c)]
    #c = c[~np.isnan(c)]
    # Get inverse closeness
    inv_c = 1/c
    return k, c, inv_c, mean_k

# split by unique values of membership
def split_kc(k, c, inv_c, mean_k, membership):
    '''
    Split kc by membership
    Parameters
    ----------
    k : array
        Degree of each node
    c : array
        Closeness of each node
    inv_c : array
        Inverse Closeness of each node
    mean_k : float
        Mean degree
    membership : array
        Membership of each node
    Returns
    -------
    ks : array
        Array of degrees for each block
    cs : array
        Array of closeness for each block
    inv_cs : array
        Array of inverse closeness for each block
    mean_ks : array
        Array of mean degree for each block
        '''
    # Get unique values of membership
    unique = np.unique(membership)
    # Get number of blocks
    num_blocks = len(unique)
    # Create arrays to store values
    ks = []
    cs = []
    inv_cs = []
    mean_ks = []
    # Loop through each block
    for i in range(num_blocks):
        # Get indices of nodes in block
        indices = np.where(membership == unique[i])
        # Get values of k, c and inv_c for nodes in block
        k_temp = k[indices]
        c_temp = c[indices]
        inv_c_temp = inv_c[indices]
        # Get mean degree of block
        mean_k_temp = np.mean(k_temp)
        # Append values to arrays
        ks.append(k_temp)
        cs.append(c_temp)
        inv_cs.append(inv_c_temp)
        mean_ks.append(mean_k_temp)
    return ks, cs, inv_cs, mean_ks

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

sizes = [1000, 1000]

ps = np.array([ [8000, 500],
                [500, 1000]])

G = collection.ns['marvel_universe']

state = gt.minimize_nested_blockmodel_dl(G)

S1 = state.entropy()

for i in range(1000): # this should be sufficiently large
    state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

S2 = state.entropy()

print("Improvement:", S2 - S1)

for i in range(1000): # this should be sufficiently large
    state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

S3 = state.entropy()

print("Improvement:", S3 - S2)

for i in range(1000): # this should be sufficiently large
    state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

S4 = state.entropy()

print("Improvement:", S4 - S3)

state.print_summary()
BS = []
levels = state.get_levels()
for s in levels:
    print(s)
    B = s.get_blocks()
    BS.append(B)
    if s.get_N() == 1:
        break

for i in range(len(BS)):
    vertices = G.get_vertices()

    mem = np.zeros(len(vertices))
    B = BS[i]
    for j in range(len(vertices)):
        mem[vertices[j]] = B[j]

    k, c, inv_c, mean_k = GetKC(G)
    #print(k)
    ks, cs, inv_cs, mean_ks = split_kc(k, c, inv_c, mean_k, mem)
    #fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    fig2, ax2 = plt.subplots(1, 1, figsize = (10, 10))
    for k in range(len(ks)):
        stat_dict = aggregate_dict(ks[k], inv_cs[k])
        ks_final, inv_c_mean, errs, stds, counts = unpack_stat_dict(stat_dict)
        #ax.errorbar(ks_final, inv_c_mean, yerr = errs, label = 'Block {}'.format(i+1))
        ax2.plot(ks[k], inv_cs[k],'x', label = 'Block {}'.format(k+1))
    '''
    ax.set_xlabel('Degree')
    ax.set_ylabel('Inverse Closeness')
    ax.set_xscale('log')
    ax.legend()
    plt.show()
    '''
    print(i)
    ax2.set_title('Level {}'.format(i+1))
    ax2.set_xlabel('Degree')
    ax2.set_ylabel('Inverse Closeness')
    ax2.set_xscale('log')
    ax2.legend()
    plt.show()
