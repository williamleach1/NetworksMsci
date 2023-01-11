from ProcessBase import *
import warnings
import scipy as sp
from Plotter import *
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


    columns =   ["N", "E", "1/ln(z)", "1/ln(z) err", "Beta", "Beta err", "rchi",
                 "pearson r", "pearson p-val", "spearmans r", "spearmans p-val", 
                 "density", "av_degree", "clustering", "L", "SWI", "asortivity", 
                 "std_degree"]

    final_df = pd.DataFrame(columns=columns)
    

    error_report = []
    num = len(names)
    pbar = tqdm((range(num)))
    for i in pbar:
        pbar.set_postfix({'Network ': names[i]})
        # Using try to catch errors
        try:
            g = load_graph(names[i])
            k, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k= process(g, to_print=False)
            a = popt[0]
            b = popt[1]
            a_err = np.sqrt(pcov[0][0])
            b_err = np.sqrt(pcov[1][1])
            plots = Plotter(names[i])
            ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
            plots.add_plot(ks,inv_c_mean,yerr=errs,fitline=True,function=Tim,popt=[a,b])
            save_name = 'Output/RealUniNets/' + names[i] + '/K_Inv_C_Clean.png'
            plots.plot(save=True,savename=save_name)

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
            num_verticies = len(g.get_vertices())
            num_edges = len(g.get_edges())
            # find densiy of network which is number of edges/number of possible edges
            density = num_edges/(num_verticies*(num_verticies-1)/2)
            # find average and standard deviation of degree
            avg_degree = mean_k
            std_degree = np.std(k)
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
            

            if SWI > 1:
                SWI = 1
            elif SWI < 0:
                SWI = 0
            temp_df = pd.DataFrame({"N": num_verticies,"E":num_edges ,"1/ln(z)": a, "1/ln(z) err": a_err,
                                    "Beta": b, "Beta err": b_err, "rchi": rchi, "pearson r": r,
                                    "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp,
                                    "density": density, "av_degree": avg_degree, "clustering": C,
                                    "L": L, "SWI": SWI, "asortivity": assortivity, "std_degree": std_degree},
                                    index=[names[i]])
            final_df = pd.concat([final_df, temp_df])
                
        # Need to handle errors otherwise code stops. This is not best practice
        # to simply skip over erro
        except OSError:
            print('OSError')
            error_report.append([names[i], ':  OSError'])
            pass
        except KeyError:
            print('KeyError')
            error_report.append([names[i], ':  HTTP Error 401: UNAUTHORIZED'])
            pass
        except RuntimeWarning:
            print('RuntimeWarning')
            error_report.append([names[i], ':  RuntimeWarning'])
            pass
        # Some devices tested have different error instead of RuntimeWarning
        except sp.optimize._optimize.OptimizeWarning:
            print('OptimizeWarning')
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
        save_name_html = 'RealUnipartiteNets_results'
        write_html(final_df, save_name_html)
    # Print Dataframe. Bit pointless as it is saved to html 
    # and is barely readable in terminal
    if to_print:
        print('Real Unipartite done')
        print(final_df)
    return final_df


# Load in already done dataframe if it exists
if os.path.exists('Output/RealUniNets/RealUniNets.pkl'):
    df_already = pd.read_pickle('Output/RealUniNets/RealUniNets.pkl')
    already_done = df_already.index.values.tolist()
else:
    already_done = []

print(already_done)    

# Load in unipartite and run for each real networks
# Need to get column names for each network from the dataframe
# Need to do after running get_networks.py
Unipartite_df = pd.read_pickle('Data/unipartite.pkl')
upper_node_limit = 50000 # takes around 1 minute per run with 50000
# Filter out num_vertices>2000000

unipartite_df = filter_num_verticies(Unipartite_df, upper_node_limit)
uni_network_names = unipartite_df.columns.values.tolist()

# Remove networks already done
uni_network_names = [x for x in uni_network_names if x not in already_done]

print(len(uni_network_names))
# Generate file system in /Output with separate folders for each network group
# Create folder for each network group (if group) and second folder for each network
MakeFolders(uni_network_names,'RealUniNets')
# Run analysis on each network
df = run_real(uni_network_names, to_html=False, to_print=True)

# Save dataframe to pickle if does not exist
# If exists, append to existing dataframe
if os.path.exists('Output/RealUniNets/RealUniNets.pkl'):
    df2 = pd.read_pickle('Output/RealUniNets/RealUniNets.pkl')
    df = pd.concat([df, df2])
    save_name_html = 'RealUnipartiteNets_results'
    write_html(df, save_name_html)
    df.to_pickle('Output/RealUniNets/RealUniNets.pkl')
else:
    df.to_pickle('Output/RealUniNets/RealUniNets.pkl')

end = time.time()
print('Time taken: ', end-start)