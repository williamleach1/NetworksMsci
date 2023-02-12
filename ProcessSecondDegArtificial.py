from ProcessBase import *
import warnings
import scipy as sp
from archive.Plotter import *
warnings.filterwarnings("error")
import graph_tool.all as gt

start = time.time()
def run_artificial(g):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    g : graph_tool graph
    Returns
    -------     
    df : dataframe
        Dataframe containing results"""

    columns =   ["N", "E", "1/ln(z)", "1/ln(z) err", "Beta", "Beta err", "rchi",
                 "pearson r", "pearson p-val", "spearmans r", "spearmans p-val",  
                 "std_degree", "av_counts"]

    k_2, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k_2= process(g, 2, to_print=False)
    a = popt[0]
    b = popt[1]
    a_err = np.sqrt(pcov[0][0])
    b_err = np.sqrt(pcov[1][1])
    k_2s, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
    av_counts = np.mean(counts)

    num_verticies = len(g.get_vertices())
    num_edges = len(g.get_edges())

    # find average and standard deviation of degree
    avg_degree = mean_k_2
    std_degree = np.std(k_2)

    return final_df

if __name__ == "__main__":


