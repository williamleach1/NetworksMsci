from archive.Plotter import *
from ProcessBase import *
from graph_tool import generation, centrality

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

    g = generation.generate_sbm(membership, ps)#, micro_degs=True)
    return g, membership


def GetKC_membership(g, membership):
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
    membership = membership[k > 0]
    k = k[k > 0]
    # remove values if closeness is nan
    k = k[~np.isnan(c)]
    membership = membership[~np.isnan(c)]
    c = c[~np.isnan(c)]
    # Get inverse closeness
    inv_c = 1/c
    return k, c, inv_c, mean_k, membership

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


sizes = [15000, 2000]

ps = np.array([ [200, 50000],
                [50000, 200]])

g, mem = SBM(sizes, ps)

# Get degree and closeness
k, c, inv_c, mean_k,mem = GetKC_membership(g, mem)

# Split by membership
ks, cs, inv_cs, mean_ks = split_kc(k, c, inv_c, mean_k, mem)

figs, axs = plt.subplots(1, 2, figsize=(10, 5))
for i in range(len(ks)):
    axs[0].plot(ks[i], inv_cs[i],'.', label='Block {}'.format(i), alpha=0.5)
axs[0].legend()
axs[0].set_xlabel(r'$k$')
axs[0].set_ylabel(r'$\frac{1}{c}$')
axs[0].set_xscale('log')


# Aggregate over k to find mean and standard error in inv_c
# for each value of k
for i in range(len(ks)):
    stat_dict = aggregate_dict(ks[i], inv_cs[i])
    ks_final, inv_c_mean, errs, stds, counts = unpack_stat_dict(stat_dict)
    axs[1].errorbar(ks_final, inv_c_mean, yerr=errs, fmt='none', label='Block {}'.format(i))
axs[1].legend()
axs[1].set_xlabel(r'$k$')
axs[1].set_ylabel(r'$\frac{1}{c}$')
axs[1].set_xscale('log')
plt.show()


