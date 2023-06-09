from ProcessBase import *
import warnings
import scipy as sp

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
def run_real(names):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    name : string
        Name of graph
    Returns
    -------     
    df : dataframe
        Dataframe containing results"""
    final_df = pd.DataFrame(columns=['mean_k_1', 'mean_k_2', 'rchi_1', 
                                'rchi_2','grad_1','grad_1_error','grad_2','grad_2_error', 
                                'gamma_1','gamma_2','rchi_uni','N_1','N_2'])
    error_report = []
    num = len(names)
    pbar = tqdm((range(num)))
    for i in pbar:
        pbar.set_postfix({'Network ': names[i]})
        # Using try to catch errors
        try:
            g = load_graph(names[i])
            if '/' in names[i]:
                name = names[i].split('/')
                name = name[0]+'_'+name[1]
            else:
                name = names[i]

            output = process_bipartite_uncon(g,Real = True, Name=name)
            # Not a fan of having all these variables
            k_1 = output[0]
            N_1 =len(k_1)
            c_1 = output[1]
            inv_c_1 = output[2]
            k_2 = output[3]
            N_2 = len(k_2)
            c_2 = output[4]
            inv_c_2 = output[5]
            mean_k_1 = output[6]
            mean_k_2 = output[7]
            rchi_1 = output[8]
            rchi_2 = output[9]
            popt_1 = output[10]
            pcov_1 = output[11]
            popt_2 = output[12]
            pcov_2 = output[13]

            grad_1 = popt_1[0]
            grad_2 = popt_2[0]

            grad_1_err = np.sqrt(pcov_1[0][0])
            grad_2_err = np.sqrt(pcov_2[0][0])

            gamma_1 = popt_1[1]
            gamma_2 = popt_2[1]


            statistics_dict_1 = output[14]
            statistics_dict_2 = output[15]
            ks_1, inv_c_mean_1, errs_1, stds_1, counts_1 = unpack_stat_dict(statistics_dict_1)
            ks_2, inv_c_mean_2, errs_2, stds_2, counts_2 = unpack_stat_dict(statistics_dict_2)

            folder = 'Output/RealBipartiteNets/'+bipartite_network_names[i]+'/'

            k_uni, c_uni,popt_uni,pcov_uni, rchi_uni, r, rp, rs, rsp, statistics_dict_uni, mean_k_uni= process(g, 1, Real = True, Name = name )
            rchi_uni = calculate_r_squared(k_uni, 1/c_uni, Tim, popt_uni)
            
            inv_c_uni = 1/c_uni
            a_uni = popt_uni[0]
            b_uni = popt_uni[1]
            a_err_uni = np.sqrt(pcov_uni[0][0])
            b_err_uni = np.sqrt(pcov_uni[1][1])
            ks_uni, inv_c_mean_uni, errs_uni, stds_uni, counts_uni   = unpack_stat_dict(statistics_dict_uni)

            av_counts = np.mean(counts_uni)           

            plt.figure()
            # Plot combined
            plt.errorbar(ks_uni, inv_c_mean_uni, yerr=errs_uni, fmt='.' ,markersize = 5,capsize=2,color='black', alpha=0.5)
            plt.plot(ks_uni, inv_c_mean_uni,'go', label="Combined mean", alpha=0.8)

            # Plot group 1
            plt.errorbar(ks_1, inv_c_mean_1, yerr=errs_1, fmt='.' ,markersize = 5,capsize=2,color='black', alpha=0.5)
            plt.plot(ks_1, inv_c_mean_1,'ro', label="Group A mean", alpha=0.8)

            # Plot group 2
            plt.errorbar(ks_2, inv_c_mean_2, yerr=errs_2, fmt='.' ,markersize = 5,capsize=2,color='black', alpha=0.5)
            plt.plot(ks_2, inv_c_mean_2,'bo', label="Group B mean", alpha=0.8)


            # Plot fit for both groups + combined
            plt.plot(k_1, Tim(k_1, *popt_1),'r--', label="Group A fit")
            plt.plot(k_2, Tim(k_2, *popt_2),'b--', label="Group B fit")
            plt.plot(k_uni, Tim(k_uni, *popt_uni),'k--', label="Combined fit")
            plt.legend(fontsize = 30)
            plt.xlabel(r"$k$", fontsize=30)
            plt.ylabel(r"$\dfrac{1}{c}$", fontsize=30, rotation=0, labelpad=30)
            plt.xscale("log")
            plt.suptitle(bipartite_network_names[i])
            rchi_latex = r'$\chi^2_{r}$'
            #plt.title('Combined '+ rchi_latex + ' = ' + str(np.round(rchi_uni,3)) + ', Group A '+ rchi_latex + ' = ' + str(np.round(rchi_1,3)) + ', Group B '+ rchi_latex + ' = ' + str(np.round(rchi_2,3)), fontsize=25)

            # Also save in plots folder (gets messy but easy to view many plots)
            plt.savefig('plots/'+str(np.round(rchi_1,3))+'_'+str(np.round(rchi_2,3))+'inv_c_vs_k.png')
            plt.savefig(folder+'inv_c_vs_k_full_fit_uncon.png')
            plt.close()
            

            # Get into dataframe to save results
            temp_df = pd.DataFrame({'mean_k_1': [mean_k_1], 'mean_k_2': [mean_k_2], 'rchi_1': [rchi_1], 
                                'rchi_2': [rchi_2], 'grad_1': [grad_1], 'grad_1_error': [grad_1_err], 
                                'grad_2': [grad_2], 'grad_2_error': [grad_2_err], 'gamma_1': [gamma_1], 'gamma_2': [gamma_2],
                                'rchi_uni': [rchi_uni],'N_1':[N_1],'N_2':[N_2]},index=[names[i]])
            final_df = pd.concat([final_df, temp_df])
        # Need to handle errors otherwise code stops. This is not best practice
        # to simply skip over error but we display failed networks at the end.
        
        except OSError:
            error_report.append([names[i], ':  OSError'])
            pass
        except KeyError:
            error_report.append([names[i], ':  HTTP Error 401: UNAUTHORIZED'])
            pass
        except RuntimeWarning:
            error_report.append([names[i], ':  RuntimeWarning'])
            pass
        #Some devices tested have different error instead of RuntimeWarning
        except sp.optimize._optimize.OptimizeWarning:
            error_report.append([names[i], ':  OptimizeWarning'])
            pass 
       
    # Printing error report
    print('-----------------------------------')
    print('Error report: \n')
    for i in error_report:
        print(i[0], i[1])
    print('-----------------------------------')
    
    return final_df

# Load in unipartite and run for each real networks
# Need to get column names for each network from the dataframe
# Need to do after running get_networks.py
Bipartite_df = pd.read_pickle('Data/bipartite.pkl')
upper_node_limit = 300000 # takes around 1 minute per run with 50000
# Filter out num_vertices>2000000

Bipartite_df = filter_num_verticies(Bipartite_df, upper_node_limit)
bipartite_network_names = Bipartite_df.columns.values.tolist()
print(len(bipartite_network_names))
# Generate file system in /Output with separate folders for each network group
# Create folder for each network group (if group) and second folder for each network
MakeFolders(bipartite_network_names,'RealBipartiteNets')
# Run analysis on each network

df = run_real(bipartite_network_names)

# path to save and load data that have been processed
save_name_df = 'Output/RealBipartiteNets/RealBipartiteNets_uncon.pkl'
df.to_pickle(save_name_df)
save_name_html = 'ReaBipartiteNets_results_uncon'
write_html(df, save_name_html)

# Measure time taken to run to help predict for larger networks
# Takes around 1-2 minute per run with 50000 upper node limit
# Takes around 40 minutes - 1 hour per run with 200000 upper node limit
# Takes around 7 hours with 500000 upper node limit
end = time.time()
print('Time taken: ', end-start)