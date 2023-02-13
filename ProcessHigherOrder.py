from ProcessBase import *
import warnings
import scipy as sp
warnings.filterwarnings("error")
import graph_tool.all as gt
from graph_tool import correlations, generation

params =    {'font.size' : 16,
            'axes.labelsize':16,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 18,
            'axes.titlesize': 16,
            'figure.titlesize': 16,
            'figure.figsize': (12, 9),}


start = time.time()
def run_real(names, to_html=False):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    name : string
        Name of graph
    Returns
    -------     
    df : dataframe
        Dataframe containing results"""

    columns =   ["N", "E", "1/ln(z)", "1/ln(z) err", "Beta", "Beta err", "rchi",
                 "pearson r", "pearson p-val", "spearmans r", "spearmans p-val", 
                 "Beta fit", "Beta Fit err" ,"density", "av_degree", "clustering", 
                 "L", "SWI", "asortivity", "std_degree", "av_counts"]

    final_df = pd.DataFrame(columns=columns)
    
    error_report = []
    num = len(names)
    pbar = tqdm((range(num)))
    for i in pbar:
        pbar.set_postfix({'Network ': names[i]})
        # Using try to catch errors
        try:
            g = load_graph(names[i])
            # Make the name by splitting at '/' if it existts and replace with '_'
            # This is to make the name of the file the same as the name of the graph
            # if the graph is loaded from a file
            if '/' in names[i]:
                name = names[i].split('/')
                name = name[0]+'_'+name[1]
            else:
                name = names[i]
            # Now process the graph
            k, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k= process(g, 3, Real = True, Name = name )
            inv_c = 1/c
            a = popt[0]
            b = popt[1]
            a_err = np.sqrt(pcov[0][0])
            b_err = np.sqrt(pcov[1][1])
            #print(popt)
            ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
            av_counts = np.mean(counts)

            # Now for aggregated data plots

            plt.figure()
            plt.title(names[i])
            folder = 'Output/RealUniNets/' + names[i] + '/'
            # Uncomment to plot all (unaggregated) points
            plt.plot(k, inv_c,'r.', label="Group 1", alpha=0.1)
            plt.xscale("log")
            plt.xlabel(r"$k$")
            plt.ylabel(r"$\frac{1}{c}$", rotation=0)
            plt.savefig(folder + 'inv_c_vs_k_unagg_HO.svg', dpi=900)
            plt.close()
            # Now for aggregated data plots
            plt.figure()
            plt.errorbar(ks, inv_c_mean, yerr=errs, fmt='.' ,markersize = 5,capsize=2,color='black')
            plt.plot(ks, inv_c_mean,'ro', label="Data")
            # Plot fit
            plt.plot(ks, HO_Tim(ks, *popt),'b--', label="Fit to data")
            plt.legend()
            plt.xlabel(r"$k$")
            plt.ylabel(r"$\frac{1}{c}$", rotation=0)
            plt.xscale("log")
            plt.title(names[i])
            plt.savefig(folder + 'inv_c_vs_k_agg_HO.svg', dpi=900)
            plt.close()

            # Now for collapse plot
            num_verticies = len(g.get_vertices())
            inv_c_pred = HO_Tim(ks,a,b,num_verticies)
            y = inv_c_mean/inv_c_pred
            y_err = errs/inv_c_pred
            plt.figure()
            plt.title(names[i])
            # Shade +/- 0.05
            plt.fill_between(ks, 1-0.05, 1+0.05, color='grey', alpha=0.2, label=r'$\pm 5\%$')
            plt.errorbar(ks, y, yerr=y_err, fmt='.' ,markersize = 5,capsize=2,color='black')
            plt.plot(ks, y,'ro', label = 'Data')
            plt.xlabel("k")
            plt.ylabel(r"$\frac{\hat{c}}{c}$", rotation=0)
            plt.legend()
            plt.savefig(folder + 'inv_c_vs_k_collapse_HO.svg', dpi=900)
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
            plt.fill_between(1/inv_c_pred, 1/inv_c_pred*(1-0.05), 1/inv_c_pred*(1+0.05), color='grey', alpha=0.2, label=r'$\pm 5\%$')

            plt.savefig(folder + 'c_vs_c_hat_HO.svg', dpi=900)
            plt.close()

            # Now for network stats
            num_edges = len(g.get_edges())

            # find average and standard deviation of degree
            avg_degree = mean_k
            std_degree = np.std(k)
            # find average path length
            L = np.sum(1/c) / (2*num_verticies)



            # Get value of beta from fit z
            try:
                Bfit, Bfit_err = beta_fit(a, a_err, num_verticies)
            except ValueError:
                Bfit = 0
                Bfit_err = 0
                pass
            except OverflowError:
                Bfit = 0
                Bfit_err = 0
                pass
            # find densiy of network which is number of edges/number of possible edges
            density = num_edges/(num_verticies*(num_verticies-1)/2)
            
            # find average clustering coefficient
            (C,var_c) = gt.global_clustering(g)
            # find average path length
            L = np.sum(1/c) / (2*num_verticies)
            # find assortativity
            assortivity, variance = correlations.assortativity(g, "total")
            # calculate small worldness
            c_r = avg_degree / num_verticies
            c_l = 3 * ( avg_degree - 2 ) / ( 4 * (avg_degree - 1) )
            L_r = np.log(num_verticies)/np.log(avg_degree)
            L_l = num_verticies / (2* avg_degree)            
            # calculate small worldness
            SWI = (( L - L_l ) / ( L_r - L_l )) * (( C - c_r ) / ( c_l - c_r ))

            # make sure SWI is between 0 and 1 (currently broken ^)
            if SWI > 1:
                SWI = 1
            elif SWI < 0:
                SWI = 0

            # Calculate Bipartivity
            
            # Calculate Treeness

            # Calculate power law prob (or most likely degree distribution)
            
            temp_df = pd.DataFrame({"N": num_verticies,"E":num_edges ,"1/ln(z)": a, "1/ln(z) err": a_err,
                                    "Beta": b, "Beta err": b_err, "rchi": rchi, "pearson r": r,
                                    "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp,
                                    "Beta fit": Bfit, "Beta Fit err": Bfit_err, "density": density, 
                                    "av_degree": avg_degree, "clustering": C,"L": L, "SWI": SWI, 
                                    "asortivity": assortivity, "std_degree": std_degree,
                                    "av_counts": av_counts}, index=[names[i]])
            final_df = pd.concat([final_df, temp_df])
                
        # Need to handle errors otherwise code stops. This is not best practice
        # to simply skip over erro
        except OSError:
            error_report.append([names[i], ':  OSError'])
            pass
        except KeyError:
            error_report.append([names[i], ':  HTTP Error 401: UNAUTHORIZED'])
            pass
        except RuntimeWarning:
            error_report.append([names[i], ':  RuntimeWarning'])
            pass
        # Some devices tested have different error instead of RuntimeWarning
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
Unipartite_df = pd.read_pickle('Data/unipartite.pkl')
upper_node_limit = 100000 # takes around 1 minute per run with 50000
# Filter out num_vertices>2000000

unipartite_df = filter_num_verticies(Unipartite_df, upper_node_limit)
uni_network_names = unipartite_df.columns.values.tolist()

# Generate file system in /Output with separate folders for each network group
# Create folder for each network group (if group) and second folder for each network
MakeFolders(uni_network_names,'RealUniNets')
# Run analysis on each network
df = run_real(uni_network_names, to_html=False )


# Save dataframe to html and pickle
save_name_html = 'RealUnipartiteNets_HO_results'
df.to_pickle('Output/RealUniNets/RealUniNets_HO.pkl')
write_html(df, save_name_html)

end = time.time()
print('Time taken: ', end-start)