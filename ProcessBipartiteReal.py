from ProcessBase import *
import warnings
import scipy as sp
from Plotter import *
warnings.filterwarnings("error")



start = time.time()
def run_real(names,to_html=False, to_print=False,func=Harry):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    name : string
        Name of graph
    Returns
    -------     
    df : dataframe
        Dataframe containing results"""
    final_df = pd.DataFrame(columns=["N","a", "a err", "b",
                                "b err", "c", "c err", "rchi", "pearson r","pearson p-val",
                                "spearmans r","spearmans p-val"])
    error_report = []
    num = len(names)
    pbar = tqdm((range(num)))
    for i in pbar:
        pbar.set_postfix({'Network ': names[i]})
        # Using try to catch errors
        try:
            g = load_graph(names[i])
            k, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k= process(g,func, 
                                                                                            to_print=False)
            a = popt[0]
            b = popt[1]
            x = popt[2]
            a_err = np.sqrt(pcov[0][0])
            b_err = np.sqrt(pcov[1][1])
            x_err = np.sqrt(pcov[2][2])
            plots = Plotter(names[i])
            ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
            plots.add_plot(ks,inv_c_mean,yerr=errs,fitline=True,function=func,popt=[a,b,x])
            save_name = 'Output/RealBipartiteNets/' + names[i] + '/K_Inv_C.png'
            plots.plot(save=True,savename=save_name)
            # now for unaggragated data plots
            plots_unag = Plotter(names[i])
            plots_unag.add_plot(k,1/c,fitline=True,function=func,popt=[a,b,x])  
            save_name2 = 'Output/RealBipartiteNets/' + names[i] + '/K_Inv_C_unagg.png'
            plots_unag.plot(save=True,savename=save_name2)
            # Now for collapse plot
            plots_collapse1 = Plotter(names[i])
            inv_c_pred = func(ks,a,b,x)
            plots_collapse1.add_plot(ks,inv_c_mean/inv_c_pred,yerr=errs/inv_c_pred)
            save_name3 = 'Output/RealBipartiteNets/' + names[i] + '/K_Inv_C_collapse1.png'
            plots_collapse1.plot(save=True,savename=save_name3)
            temp_df = pd.DataFrame({"N": len(g.get_vertices()), "a": a, "a err": a_err, 
                                "b": b, "b err": b_err,"c":x,"c err":x_err, "rchi": rchi, "pearson r": r,
                                "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp}, 
                                index=[names[i]])
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
upper_node_limit = 3000 # takes around 1 minute per run with 50000
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