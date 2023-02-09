from ProcessBase import *
import warnings
import scipy as sp
from Plotter import *
warnings.filterwarnings("error")
import graph_tool.all as gt
from graph_tool import correlations, generation


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
                 "density", "av_degree", "clustering", "L", "SWI", "asortivity", 
                 "std_degree", "av_counts"]

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
            k, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k= process(g, 1, Real = True, Name = name )
            a = popt[0]
            b = popt[1]
            a_err = np.sqrt(pcov[0][0])
            b_err = np.sqrt(pcov[1][1])
            plots = Plotter(names[i])
            ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
            av_counts = np.mean(counts)

            # Now for aggregated data plots
            plots.add_plot(ks,inv_c_mean,yerr=errs,fitline=True,function=Tim,popt=[a,b])
            save_name = 'Output/RealUniNets/' + names[i] + '/K_Inv_C_Clean.png'
            plt.savefig(save_name)

            # now for unaggragated data plots
            plots_unag = Plotter(names[i])
            plots_unag.add_plot(k,1/c,fitline=True,function=Tim,popt=[a,b])
            save_name2 = 'Output/RealUniNets/' + names[i] + '/K_Inv_C_unagg_clean.png'
            plots_unag.plot(save=True,savename=save_name2)

            # Now for collapse plot
            plots_collapse1 = Plotter(names[i])
            inv_c_pred = Tim(ks,a,b)
            plots_collapse1.add_plot(ks,inv_c_mean/inv_c_pred,yerr=errs/inv_c_pred)
            save_name3 = 'Output/RealUniNets/' + names[i] + '/K_Inv_C_collapse1_clean.png'
            plots_collapse1.plot(save=True,savename=save_name3)

            # find average and standard deviation of degree
            avg_degree = mean_k
            std_degree = np.std(k)
            # find average path length
            L = np.sum(1/c) / (2*num_verticies)

            # Now for network stats
            num_verticies = len(g.get_vertices())
            num_edges = len(g.get_edges())
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
                                    "density": density, "av_degree": avg_degree, "clustering": C,
                                    "L": L, "SWI": SWI, "asortivity": assortivity, "std_degree": std_degree,
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
upper_node_limit = 70000 # takes around 1 minute per run with 50000
# Filter out num_vertices>2000000

unipartite_df = filter_num_verticies(Unipartite_df, upper_node_limit)
uni_network_names = unipartite_df.columns.values.tolist()

# Generate file system in /Output with separate folders for each network group
# Create folder for each network group (if group) and second folder for each network
MakeFolders(uni_network_names,'RealUniNets')
# Run analysis on each network
df = run_real(uni_network_names, to_html=False )


# Save dataframe to html and pickle
save_name_html = 'RealUnipartiteNets_results'
df.to_pickle('Output/RealUniNets/RealUniNets.pkl')
write_html(df, save_name_html)

end = time.time()
print('Time taken: ', end-start)