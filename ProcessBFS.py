from ProcessBase import *
import warnings
import scipy as sp
warnings.filterwarnings("error")
import graph_tool.all as gt
from graph_tool import correlations, generation
from uncertainties import ufloat, umath

params =    {'font.size' : 16,
            'axes.labelsize':16,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 18,
            'axes.titlesize': 16,
            'figure.titlesize': 16,
            'figure.figsize': (12, 12),}
plt.rcParams.update(params)

start = time.time()

def L_nk(ks,N,z):
    """Calculate L_nk
    Parameters
    ----------
    k : int
        Degree
    N : int
        Number of nodes
    z : float
        Scaling parameter
    Returns
    -------
    L_nk : float
        L_nk"""
    L = np.log(N*(z-1)/ks)/np.log(z)
    return L

def n_l(l,z,av_k,L):
    """Calculate n_l
    Parameters
    ----------
    l : int
        Distance
    z : float
        Scaling parameter
    k : int
        Degree
    L : float
        L_nk
    Returns
    -------
    n_l : float
        n_l"""

    n = (l<L) * av_k*np.power(z,(l-1),where=l<L)
    return n
    

def run_bfs(g, Real, Name = None):
    """Perform all analysis on graph
    Parameters""" 
    
    
    # Make the name by splitting at '/' if it existts and replace with '_'
    # This is to make the name of the file the same as the name of the graph
    # if the graph is loaded from a file
    if Real:
        if '/' in Name:
            name = Name.split('/')
            name = name[0]+'_'+name[1]
        else:
            name = Name
    else:
        name = Name
    # Now process the graph
    k, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k= process(g, 1, Real = True, Name = name )
    inv_c = 1/c
    inv_ln_z = popt[0]
    beta = popt[1]
    inv_ln_z_err = np.sqrt(pcov[0][0])
    beta_err = np.sqrt(pcov[1][1])
    inv_ln_z_u = ufloat(inv_ln_z, inv_ln_z_err)
    beta_u = ufloat(beta, beta_err)

    ln_z_u = 1/inv_ln_z_u
    z_u = umath.exp(ln_z_u)
    z = z_u.n
    z_err = z_u.s

    Ls = L_nk(k, g.num_vertices(), z)
    Mean_L = np.mean(Ls)
    Mean_L_err = np.std(Ls)#/np.sqrt(len(Ls))

    max_z = z + z_err
    min_z = z - z_err

    # Now get the BFS results
    unq_dist, mean_count, std_count, err_count  = process_BFS(g, Real = True, Name = name)
    av_k = np.mean(k)
    av_k_err = np.std(k)#/np.sqrt(len(k))
    dist = np.linspace(0, max(unq_dist)+1,100)
    
    ns = n_l(dist, z, av_k, Mean_L)
    max_ns = n_l(dist, max_z, av_k+av_k_err, Mean_L+Mean_L_err)
    min_ns = n_l(dist, min_z, av_k-av_k_err, Mean_L-Mean_L_err)

    # Now plot the results
    fig, ax = plt.subplots(1,1)
    ax.errorbar(unq_dist, mean_count, yerr = std_count, fmt = 'o', capsize = 5,color='Red', label = 'Numerical')
    ax.plot(dist, ns,'k--' , label = 'From Fit')
    ax.fill_between(dist, min_ns, max_ns,color='Blue', alpha = 0.2, label = 'Fit from '+ r'$\pm 1\sigma$')
    ax.set_xlabel(r'$l$', fontsize = 30, labelpad = 20)
    ax.set_ylabel(r'$n(l)$', fontsize = 30,rotation = 0, labelpad = 20)
    #ax.set_title('BFS for {}'.format(Name))
    #ax.set_title('ER with 10000 nodes and Average degree = 10')
    ax.legend(fontsize = 25)
    '''
    ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
    ax[1].errorbar(ks, inv_c_mean, yerr = errs, fmt = 'o', label = 'Data')
    ax[1].plot(ks, Tim(ks, *popt), 'k--', label = 'Fit')
    ax[1].set_xlabel('Degree')
    ax[1].set_ylabel('Inverse Closeness')
    ax[1].set_title('Inverse Closeness for {}'.format(Name))
    ax[1].legend()
    ax[1].set_xscale('log')
    fig.suptitle('z = {} +/- {}, rchi = {}'.format(z, z_err, rchi))
    '''
    if Real:
        folder = 'Output/RealUniNets/' + Name + '/'
    else:
        folder = 'Output/ArtificialUniNets/'+Name+'/'
    fig.savefig(folder+'bfs.svg', dpi=900)
    plt.close()

if __name__ == '__main__':
    # Load in unipartite and run for each real networks
    # Need to get column names for each network from the dataframe
    # Need to do after running get_networks.py
    Unipartite_df = pd.read_pickle('Data/unipartite.pkl')
    upper_node_limit = 10000 # takes around 1 minute per run with 50000
    # Filter out num_vertices>2000000

    unipartite_df = filter_num_verticies(Unipartite_df, upper_node_limit)
    uni_network_names = unipartite_df.columns.values.tolist()

    # Generate file system in /Output with separate folders for each network group
    # Create folder for each network group (if group) and second folder for each network
    MakeFolders(uni_network_names,'RealUniNets')
    # Run analysis on each network
    '''
    for i in uni_network_names:
        try:
            g = load_graph(i)
            run_bfs(g, Real = True, Name = i) 
        except RuntimeWarning:
            print('RuntimeWarning for', i)
            pass
        except OSError:
            print('OSError for', i)
            pass
        except ValueError:
            print('ValueError for', i)
            pass
    '''
    g = ER(10000,10)
    run_bfs(g,False,'ER') 
