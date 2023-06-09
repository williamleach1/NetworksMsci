from ProcessBase import *
import warnings
import scipy as sp
from archive.Plotter import *
warnings.filterwarnings("error")
import graph_tool.all as gt
from graph_tool import correlations, generation
from copy import deepcopy


start = time.time()
def run_real(names, to_html=False, to_print=False):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    name : string
        Name of graph
    Returns
    -------     
    df : dataframe
        Dataframe containing results"""


    columns =   ["N","E","1/ln(z)","1/ln(z) err","Gamma","Gamma err", "rchi_second",
                "av_second_degree", "std_second_degree","av_counts_second_degree"]

    final_df = pd.DataFrame(columns=columns)
    

    error_report = []
    num = len(names)
    pbar = tqdm((range(num)))
    for i in pbar:
        pbar.set_postfix({'Network ': names[i]})
        # Using try to catch errors
        try:
            # Make the name by splitting at '/' if it existts and replace with '_'
            # This is to make the name of the file the same as the name of the graph
            # if the graph is loaded from a file
            if '/' in names[i]:
                name = names[i].split('/')
                name = name[0]+'_'+name[1]
            else:
                name = names[i]            
            g = load_graph(names[i])
            k_2, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k_2= process(g, 2, Real = True, Name = name )
            a = popt[0]
            b = popt[1]
            a_err = np.sqrt(pcov[0][0])
            b_err = np.sqrt(pcov[1][1])
            k_2s, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
            av_counts = np.mean(counts)

            inv_c = 1/c
            plt.figure()
            plt.title(names[i])
            folder = 'Output/RealUniNets/' + names[i] + '/'
            # Uncomment to plot all (unaggregated) points
            plt.plot(k_2, inv_c,'r.', alpha=0.1)
            plt.xscale("log")
            plt.xlabel(r"$k_{2}$")
            plt.ylabel(r"$\frac{1}{c}$", rotation=0)
            plt.savefig(folder + 'inv_c_vs_k2_unagg.svg', dpi=900)
            plt.close()

            # Now for aggregated data plots
            plt.figure()
            plt.errorbar(k_2s, inv_c_mean, yerr=errs, fmt='.' ,markersize = 5,capsize=2,color='black')
            plt.plot(k_2s, inv_c_mean,'ro', label="Data")
            # Plot fit
            plt.plot(k_2s, Tim_2(k_2s, *popt),'b--', label="Fit to data")
            plt.legend()
            plt.xlabel(r"$k_{2}$")
            plt.ylabel(r"$\frac{1}{c}$", rotation=0)
            plt.xscale("log")
            plt.title(names[i])
            plt.savefig(folder + 'inv_c_vs_k2_agg.svg', dpi=900)
            plt.close()

            # Now for collapse plot
            inv_c_pred = Tim_2(k_2s,a,b)
            y = inv_c_mean/inv_c_pred
            y_err = errs/inv_c_pred
            plt.figure()
            plt.title(names[i])
            # Shade +/- 0.05
            plt.fill_between(k_2s, 1-0.05, 1+0.05, color='grey', alpha=0.2, label=r'$\pm 5\%$')
            plt.errorbar(k_2s, y, yerr=y_err, fmt='.' ,markersize = 5,capsize=2,color='black')
            plt.plot(k_2s, y,'ro', label = 'Data')
            plt.xlabel("k")
            plt.ylabel(r"$\frac{\hat{c}}{c}$", rotation=0)
            plt.legend()
            plt.savefig(folder + 'inv_c_vs_k2_collapse.svg', dpi=900)
            plt.close()

            plt.figure()
            plt.title(names[i])
            c_mean = 1/inv_c_mean
            c_pred = 1/inv_c_pred
            plt.plot(c_pred, c_mean, 'ro')
            plt.ylabel(r"$c$", rotation=0)
            plt.xlabel(r"$\hat{c}$", rotation=0)
            # Add line at 45 degrees and shade +/- 5%
            plt.plot(c_pred, c_pred, 'k--', label='Expected')
            plt.fill_between(1/inv_c_pred, 1/inv_c_pred*(1-0.05), 1/inv_c_pred*(1+0.05), 
                                color='grey', alpha=0.2, label=r'$\pm 5\%$')
            plt.savefig(folder + 'c_vs_c_hatK2.svg', dpi=900)
            plt.close('all')

            num_verticies = len(g.get_vertices())
            num_edges = len(g.get_edges())

            # find average and standard deviation of degree
            avg_degree = mean_k_2
            std_degree = np.std(k_2)

            temp_df = pd.DataFrame({"N": num_verticies,"E":num_edges ,"1/ln(z)": a, "1/ln(z) err": a_err,
                                    "Gamma": b, "Gamma err": b_err, "rchi_second": rchi,
                                    "av_second_degree": avg_degree, "std_second_degree": std_degree,
                                    "av_counts_second_degree": av_counts}, index=[names[i]])
            final_df = pd.concat([final_df, temp_df])
                
        # Need to handle errors otherwise code stops. This is not best practice
        # to simply skip over erro
        except OSError:
            #print('OSError')
            error_report.append([names[i], ':  OSError'])
            pass
        except KeyError:
            #print('KeyError')
            error_report.append([names[i], ':  HTTP Error 401: UNAUTHORIZED'])
            pass
        except RuntimeWarning:
            #print('RuntimeWarning')
            error_report.append([names[i], ':  RuntimeWarning'])
            pass
        # Some devices tested have different error instead of RuntimeWarning
        except sp.optimize._optimize.OptimizeWarning:
            #print('OptimizeWarning')
            error_report.append([names[i], ':  OptimizeWarning'])
            pass 
    # Printing error report
    print('-----------------------------------')
    print('Error report: \n')
    for i in error_report:
        print(i[0], i[1])
    print('-----------------------------------')
    return final_df

if __name__ == "__main__":

    # Load in unipartite and run for each real networks
    # Need to get column names for each network from the dataframe
    # Need to do after running get_networks.py
    Unipartite_df = pd.read_pickle('Data/unipartite.pkl')
    
    upper_node_limit = 20000 # takes too long to run for large networks
    
    unipartite_df = filter_num_verticies(Unipartite_df, upper_node_limit)
    uni_network_names = unipartite_df.columns.values.tolist()

    # Generate file system in /Output with separate folders for each network group
    # Create folder for each network group (if group) and second folder for each network
    MakeFolders(uni_network_names,'RealUniNets')
    # Run analysis on each network
    df = run_real(uni_network_names)

    # Save dataframe to pickle and HTML
    df.to_pickle('Output/RealUniNets/RealUniNets_K2.pkl')
    save_name_html = 'RealUnipartiteNets_2ndDeg_results'
    write_html(df, save_name_html)

    end = time.time()
    print('Time taken: ', end-start)