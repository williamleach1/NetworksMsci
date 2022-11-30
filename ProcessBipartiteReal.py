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
def run_real(names,to_html=False, to_print=False):
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
    num = len(names)
    pbar = tqdm((range(num)))
    for i in pbar:
        pbar.set_postfix({'Network ': names[i]})
        # Using try to catch errors
        try:
            g = load_graph(names[i])
            output = process_bipartite(g, to_print=True)
            
            # plot the odd and even
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
            #plt.plot(k_1, inv_c_1,'r.', label="Group 1", alpha=0.1)
            #plt.plot(k_2, inv_c_2,'b.', label="Group 2", alpha=0.1)

            plt.errorbar(ks_1, inv_c_mean_1, yerr=errs_1, fmt='.' ,markersize = 5,capsize=2,color='black')
            plt.plot(ks_1, inv_c_mean_1,'ro', label="Group 1 mean")


            plt.errorbar(ks_2, inv_c_mean_2, yerr=errs_2, fmt='.' ,markersize = 5,capsize=2,color='black')
            plt.plot(ks_2, inv_c_mean_2,'bo', label="Group 2 mean")

            plt.plot(k_1, Harry_1(k_1, *popt),'r--', label="Group 1 fit")
            plt.plot(k_2, Harry_2(k_2, *popt),'b--', label="Group 2 fit")
            plt.legend()
            plt.xlabel("k")
            plt.ylabel("1/c")
            plt.xscale("log")
            plt.suptitle(bipartite_network_names[i])
            plt.title("a = %2f, b = %2f, alpha = %2f, rchi1=%2f, rchi2=%2f" % (popt[0], popt[1], popt[2], rchi_1, rchi_2))
            
            folder = 'Output/RealBipartiteNets/'+bipartite_network_names[i]+'/'

            plt.savefig('plots/'+str(np.round(rchi_1,3))+'_'+str(np.round(rchi_2,3))+'inv_c_vs_k.png')
            plt.savefig(folder+'inv_c_vs_k_full_fit.png')
            temp_df = pd.DataFrame({'mean k 1:': [mean_k_1], 'mean k 2:': [mean_k_2], 'rchi 1:': [rchi_1], 
                                'rchi 2:': [rchi_2], 'r 1:': [r1], 'r 2:': [r2], 'rs 1:': [rs1], 
                                'rs 2:': [rs2], 'rp 1:': [rp1], 'rp 2:': [rp2], 'rsp 1:': [rsp1],
                                'rsp 2:': [rsp2], 'a:': [popt[0]], 'a error:': [errs[0]], 'b:': [popt[1]], 
                                'b error:': [errs[1]], 'alpha:': [popt[2]], 'alpha error:': [errs[2]]})
            final_df = pd.concat([final_df, temp_df])
        # Need to handle errors otherwise code stops. This is not best practice
        # to simply skip over erro
        
        except OSError:
            error_report.append([names[i], ':  OSError'])
            pass
        except KeyError:
            error_report.append([names[i], ':  HTTP Error 401: UNAUTHORIZED'])
            pass
        '''
        except RuntimeWarning:
            error_report.append([names[i], ':  RuntimeWarning'])
            pass
        # Some devices tested have different error instead of RuntimeWarning
        except sp.optimize._optimize.OptimizeWarning:
            error_report.append([names[i], ':  OptimizeWarning'])
            pass 
        '''
    # Printing error report
    print('-----------------------------------')
    print('Error report: \n')
    for i in error_report:
        print(i[0], i[1])
    print('-----------------------------------')
    # Saving dataframe to html
    if to_html:
        save_name_html = 'ReaBipartiteNets_results'
        write_html(final_df, save_name_html)
    # Print Dataframe. Bit pointless as it is saved to html 
    # and is barely readable in terminal
    if to_print:
        print('Real Bipartite done')
        print(final_df)
    return final_df

# Load in unipartite and run for each real networks
# Need to get column names for each network from the dataframe
# Need to do after running get_networks.py
Bipartite_df = pd.read_pickle('Data/bipartite.pkl')
upper_node_limit = 50000 # takes around 1 minute per run with 50000
# Filter out num_vertices>2000000
Bipartite_df = filter_num_verticies(Bipartite_df, upper_node_limit)
bipartite_network_names = Bipartite_df.columns.values.tolist()
print(len(bipartite_network_names))
# Generate file system in /Output with separate folders for each network group
# Create folder for each network group (if group) and second folder for each network
MakeFolders(bipartite_network_names,'RealBipartiteNets')
# Run analysis on each network
df = run_real(bipartite_network_names, to_html=True, to_print=True)

# Measure time taken to run to help predict for larger networks
# Takes around 1-2 minute per run with 50000 upper node limit
# Takes around 40 minutes - 1 hour per run with 200000 upper node limit
# Takes around 7 hours with 500000 upper node limit
end = time.time()
print('Time taken: ', end-start)