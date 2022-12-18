from ProcessBase import *
import warnings
import scipy as sp
from Plotter import *

warnings.filterwarnings("error")

params =    {'font.size' : 16,
            'axes.labelsize':16,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 18,
            'axes.titlesize': 16,
            'figure.titlesize': 16,
            'figure.figsize': (16, 12),}
plt.rcParams.update(params)

start = time.time()
def run_real(gen_func,all_args=[],args_mean = [],to_html=False, to_print=False):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    name : string
        Name of graph
    Returns
    -------     
    df : dataframe
        Dataframe containing results"""
    final_df = pd.DataFrame(columns=['mean k 1:', 'mean k 2:', 'rchi 1:','rchi 2:', 
                                    'r 1:', 'r 2:', 'rs 1:', 'rs 2:','rp 1:','rp 2:', 'rsp 1',
                                    'rsp 2:', 'a:', 'a error:', 'b:', 'b error:', 'alpha:', 
                                    'alpha error:'])
    error_report = []
    pbar = tqdm(range(len(all_args)))
    for i in pbar:
        
        args = all_args[i]
        pbar.set_postfix({'Network args ': args})
        # Using try to catch errors

        g = gen_func(*args)
        g = clean_graph(g)
        print('Graph generated')
        output = process_bipartite(g, to_print=False)
        
        # Not a fan of having all these variables
        k_1 = output[0]
        c_1 = output[1]
        inv_c_1 = output[2]
        k_2 = output[3]
        c_2 = output[4]
        inv_c_2 = output[5]
        mean_k_1 = output[6]
        mean_k_2 = output[7]
        rchi_1 = output[8]
        rchi_2 = output[9]
        r1 = output[10]
        r2 = output[11]
        rs1 = output[12]
        rs2 = output[13]
        rp1 = output[14]
        rp2 = output[15]
        rsp1 = output[16]
        rsp2 = output[17]
        popt = output[18]
        errs = output[19]
        statistics_dict_1 = output[20]
        statistics_dict_2 = output[21]


        ks_1, inv_c_mean_1, errs_1, stds_1, counts_1   = unpack_stat_dict(statistics_dict_1)
        ks_2, inv_c_mean_2, errs_2, stds_2, counts_2   = unpack_stat_dict(statistics_dict_2)

        plt.figure()

        # Uncomment to plot all (unaggregated) points
        #plt.plot(k_1, inv_c_1,'r.', label="Group 1", alpha=0.1)
        #plt.plot(k_2, inv_c_2,'b.', label="Group 2", alpha=0.1)

        # Plot group 1
        plt.errorbar(ks_1, inv_c_mean_1, yerr=errs_1, fmt='.' ,markersize = 5,capsize=2,color='black')
        plt.plot(ks_1, inv_c_mean_1,'ro', label="Group 1 mean")

        # Plot group 2
        plt.errorbar(ks_2, inv_c_mean_2, yerr=errs_2, fmt='.' ,markersize = 5,capsize=2,color='black')
        plt.plot(ks_2, inv_c_mean_2,'bo', label="Group 2 mean")

        # Plot fit for both groups
        plt.plot(k_1, Harry_1(k_1, *popt),'r--', label="Group 1 fit")
        plt.plot(k_2, Harry_2(k_2, *popt),'b--', label="Group 2 fit")
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("1/c")
        plt.xscale("log")
        #plt.suptitle(bipartite_network_names[i])
        plt.title("a = %2f, b = %2f, alpha = %2f, rchi1=%2f, rchi2=%2f" % (popt[0], popt[1], popt[2], rchi_1, rchi_2))
        
        # Save plot
        #folder = 'Output/RealBipartiteNets/'+bipartite_network_names[i]+'/'

        # Also save in plots folder (gets messy but easy to view many plots)
        #plt.savefig('plots/'+str(np.round(rchi_1,3))+'_'+str(np.round(rchi_2,3))+'inv_c_vs_k.png')
        #plt.savefig(folder+'inv_c_vs_k_full_fit.png')
        plt.show()
        # Get into dataframe to save results
        temp_df = pd.DataFrame({'mean k 1:': [mean_k_1], 'mean k 2:': [mean_k_2], 'rchi 1:': [rchi_1], 
                            'rchi 2:': [rchi_2], 'r 1:': [r1], 'r 2:': [r2], 'rs 1:': [rs1], 
                            'rs 2:': [rs2], 'rp 1:': [rp1], 'rp 2:': [rp2], 'rsp 1:': [rsp1],
                            'rsp 2:': [rsp2], 'a:': [popt[0]], 'a error:': [errs[0]], 'b:': [popt[1]], 
                            'b error:': [errs[1]], 'alpha:': [popt[2]], 'alpha error:': [errs[2]]})
        final_df = pd.concat([final_df, temp_df])
    # Saving dataframe to html
    if to_html:
        save_name_html = 'ArtificialBipartiteNets_results'
        write_html(final_df, save_name_html)
    # Print Dataframe. Bit pointless as it is saved to html 
    # and is barely readable in terminal
    if to_print:
        print('Artificial Bipartite done')
        print(final_df)
    return final_df

# Run the code
# First pecify names of models and arguments each take
names = ['BA_Bipartite', 'ER_bipartite']
bipartiteBA_args_mean = ['m1', 'm2', 'n']
bipartiteER_args_mean = ['n1', 'n2', 'p']

# Specify arguments for each model. Give as list of tuples for multiple runs
args_BA = [(1, 3, 30000), (2, 3, 30000)]
args_ER = [(20000, 10000, 0.0005), (20000, 8000, 0.00025)]

# Run the code for each model - specify arguments for each model and model type
run_real(BipartiteER, args_ER,bipartiteER_args_mean, to_print=True, to_html=True)

end = time.time()
print('Time taken: ', end-start)