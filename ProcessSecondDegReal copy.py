from ProcessBase import *
import warnings
import scipy as sp
from archive.Plotter import *
warnings.filterwarnings("error")
import graph_tool.all as gt
from graph_tool import correlations, generation
from copy import deepcopy
from scipy.interpolate import interp1d
import gc

plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (3.5, 2.3),
    'figure.subplot.left': 0.17,
    'figure.subplot.right': 0.98,
    'figure.subplot.bottom': 0.17,
    'figure.subplot.top': 0.98,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.5,
    'lines.markersize': 2,
})


def calc_reduced_chi_square(observed, expected, errors, num_params):
    residuals = observed - expected
    chi_square = np.sum((residuals / errors) ** 2)
    degrees_of_freedom = len(observed) - num_params
    reduced_chi_square = chi_square / degrees_of_freedom
    return reduced_chi_square

def calc_mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def calculate_r_squared(x_data, y_data, func, popt):
    """
    Calculate the R-squared value for a fitted function and data.
    
    Parameters:
    x_data : array_like
        The x data points.
    y_data : array_like
        The y data points.
    func : callable
        The fitted function.
    popt : array_like
        The optimized parameters for the fitted function.
    
    Returns:
    r_squared : float
        The R-squared value.
    """
    residuals = y_data - func(x_data, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

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


    # columns =   ["N","E","1/ln(z)","1/ln(z) err","Gamma","Gamma err", "rchi_second",
    #             "av_second_degree", "std_second_degree","av_counts_second_degree"]

    # final_df = pd.DataFrame(columns=columns)
    

    error_report = []
    num = len(names)
    pbar = tqdm((range(num)))
    r2_tims = []
    r2_HO = []
    mse_tims = []
    mse_HO = []
    nets_processed = []
    r_high_better = 0 
    mse_high_better = 0

    for i in pbar:
        pbar.set_postfix({'Network ': names[i]})
        # Using try to catch errors
        try:
            gc.collect()
            # Make the name by splitting at '/' if it existts and replace with '_'
            # This is to make the name of the file the same as the name of the graph
            # if the graph is loaded from a file
            if '/' in names[i]:
                name = names[i].split('/')
                name = name[0]+'_'+name[1]
            else:
                name = names[i]            
            g = load_graph(names[i])
            k_tim, c_tim,popt_tim,pcov_tim, rchi_tim, r_tim, rp_tim, rs_tim, rsp_tim, statistics_dict_tim, mean_k_tim= process(g, 1, Real = True, Name = name )
            ks_tim = np.unique(k_tim)
            
            inv_c_tim = 1/c_tim

            # predcted for tim
            observed_tim = inv_c_tim
            xdata_tim = k_tim
            inv_c_pred_tim = Tim(xdata_tim, *popt_tim)
            expected_tim = inv_c_pred_tim

            MSE_tim = calc_mean_squared_error(observed_tim, expected_tim)
            

            r2_tim = calculate_r_squared(k_tim, inv_c_tim, Tim, popt_tim)

            k, k_2, c, popt,pcov, statistics_dict_k,statistics_dict_k_2, mean_k_2, HO_second_degree_N_fina = process_second(g,Real = True, Name = name )
            
            all_xdata = np.vstack((k_2, k))

            all_inv_c_pred = HO_second_degree_N_fina(all_xdata, *popt)

            # Aggregate all_inv_c_pred for
            
            # Calculate the R-squared value for the fit
            k_2s, inv_c_mean_k_2, errs_k_2, stds_k_2, counts_k_2   = unpack_stat_dict(statistics_dict_k_2)
            k_1s, inv_c_mean_k_1, errs_k_1, stds_k_1, counts_k_1   = unpack_stat_dict(statistics_dict_k)
            a = popt[0]
            b = popt[1]
            a_err = np.sqrt(pcov[0][0])
            b_err = np.sqrt(pcov[1][1])

            observed = 1/c
            xdata = np.vstack((k_2, k))
            inv_c_pred = HO_second_degree_N_fina(xdata, *popt)
            expected = inv_c_pred

            MSE_HO = calc_mean_squared_error(observed, expected)

            if MSE_HO < MSE_tim:
                mse_high_better += 1

            print("MSE Tim: ", MSE_tim)
            print("MSE HO: ", MSE_HO)
            #iterate through xdata and find quadrature error from statistics_dict_k and statistics_dict_k_2
            '''
            stds = []
            new_observed = []
            new_expected = []
            for j in range(len(xdata[0])):
                k_2_val = xdata[0][j]
                k_val = xdata[1][j]
                if k_2_val in statistics_dict_k_2.keys():
                    if k_val in statistics_dict_k.keys():
                        k_2_stats = statistics_dict_k_2[k_2_val]
                        k_stats = statistics_dict_k[k_val]
                        k_2_std = k_2_stats[2]
                        k_std = k_stats[2]
                        std = np.sqrt((k_2_std)**2 + (k_std)**2)
                        if std > 0:
                            #print(std, observed[j], expected[j])
                            stds.append(std)
                            new_observed.append(observed[j])
                            new_expected.append(expected[j])
            stds = np.array(stds)
            new_observed = np.array(new_observed)
            new_expected = np.array(new_expected)
            rchi_second = calc_reduced_chi_square(new_observed, new_expected, stds, 2)
            
            print(rchi_tim, rchi_second)
            '''

            r_squared = calculate_r_squared(xdata, 1/c, HO_second_degree_N_fina, popt)
            print(r2_tim, r_squared)

            if r_squared > r2_tim:
                r_high_better += 1
            
            res = aggregate_dict(k_2, k)

            k_2_agg_a, k_agg_a, a,b,c = unpack_stat_dict(res)

            interp_func_a = interp1d(k_2_agg_a, k_agg_a, kind='linear')

            # Generate new k values for plotting the smooth function
            k2_values_smooth_a = np.linspace(k_2_agg_a.min(), k_2_agg_a.max(), 500)
            k1_values_smooth_a = interp_func_a(k2_values_smooth_a)

            new_xdata = np.vstack((k2_values_smooth_a, k1_values_smooth_a))
            inv_c_pred_k_2 = HO_second_degree_N_fina(new_xdata, *popt)
            
            res = aggregate_dict(k,k_2)

            k_agg_b,k_2_agg_b, a,b,c = unpack_stat_dict(res)

            interp_func_b = interp1d( k_agg_b,k_2_agg_b, kind='linear')

            # Generate new k values for plotting the smooth function
            k1_values_smooth_b = np.linspace(k_agg_b.min(), k_agg_b.max(), 500)
            k2_values_smooth_b = interp_func_b(k1_values_smooth_b)

            new_xdata = np.vstack((k2_values_smooth_b, k1_values_smooth_b))
            inv_c_pred_k_1 = HO_second_degree_N_fina(new_xdata, *popt)
            # res = aggregate_dict( k, k_2)
            # k_agg_b,k_2_agg_b, a,b,c = unpack_stat_dict(res)
            #new_xdata = np.vstack(( k_2_agg_b, k_agg_b))
            #inv_c_pred_k = HO_second_degree_N_fina(new_xdata, *popt)
            
            inv_c = 1/c
            folder = 'Output/RealUniNets/' + names[i] + '/'

            # Now for aggregated data plots
            plt.figure(figsize=(3.5,2.4))
            plt.errorbar(k_2s, inv_c_mean_k_2, yerr=errs_k_2, fmt='o' ,markersize = 4,capsize=2,color='red', mfc= 'none', label="Data", elinewidth=0.5, mew=1, capthick=0.5)
            #plt.plot(k_2s, inv_c_mean_k_2,'ro', label="Data")
            # Plot fit
            plt.plot(k2_values_smooth_a, inv_c_pred_k_2,'b--', label="Interpolated fit to data")
            plt.legend()
            plt.xlabel(r"$k^{(2)}$")
            plt.ylabel(r"$\dfrac{1}{c}$", rotation=0, labelpad=10)
            plt.xscale("log")
            plt.title(names[i])
            plt.savefig(folder + 'A_k2_agg_best.png', dpi=900)
            plt.close()

            plt.figure(figsize=(3.5,2.4))
            plt.errorbar(k_1s, inv_c_mean_k_1, yerr=errs_k_1, fmt='o' ,markersize = 5,capsize=2,color='red', mfc= 'none', label="Data", elinewidth=0.5, capthick=0.5, mew = 1)
            plt.plot(ks_tim, Tim(ks_tim, *popt_tim),'k--', label="Lower order Fit")
            # Plot fit
            plt.plot(k1_values_smooth_b, inv_c_pred_k_1,'b--', label="Interpolated fit to data")
            plt.legend()
            plt.xlabel(r"$k^{(1)}$") 
            plt.ylabel(r"$\dfrac{1}{c}$", rotation=0, labelpad=10)
            plt.xscale("log")
            plt.title(names[i])
            plt.savefig(folder + 'A_k1_agg_best.png', dpi=900)
            plt.close()
            '''
            plt.figure()
            plt.errorbar(k_1s, inv_c_mean_k_1, yerr=errs_k_1, fmt='.' ,markersize = 5,capsize=2,color='black')
            plt.plot(k_1s, inv_c_mean_k_1,'ro', label="Data")
            # Plot fit
            plt.plot(k_agg_b, inv_c_pred_k,'b--', label="Fit to data")
            plt.legend()
            plt.xlabel(r"$k_{2}$")
            plt.ylabel(r"$\frac{1}{c}$", rotation=0)
            plt.xscale("log")
            plt.title(names[i])
            plt.savefig(folder + 'A_k1_agg_best.png', dpi=900)
            plt.close()
            '''
            '''
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
            '''
            r2_HO.append(r_squared)
            mse_HO.append(MSE_HO)
            r2_tims.append(r2_tim)
            mse_tims.append(MSE_tim)
            nets_processed.append(names[i])
            del statistics_dict_k_2
            del statistics_dict_k
            del k_2s
            del inv_c_mean_k_2
            del errs_k_2
            del stds_k_2
            del counts_k_2
            del k_1s
            del inv_c_mean_k_1
            del errs_k_1
            del stds_k_1
            del counts_k_1
            del inv_c
            del inv_c_pred_k_2
            del inv_c_pred_k_1
            del inv_c_pred

            num_verticies = len(g.get_vertices())
            num_edges = len(g.get_edges())
            del g
            # find average and standard deviation of degree
            avg_degree = mean_k_2
            std_degree = np.std(k_2)

            # temp_df = pd.DataFrame({"N": num_verticies,"E":num_edges ,"1/ln(z)": a, "1/ln(z) err": a_err,
            #                         "Gamma": b, "Gamma err": b_err, "rchi_second": 0,
            #                         "av_second_degree": avg_degree, "std_second_degree": std_degree,
            #                         }, index=[names[i]])
            # final_df = pd.concat([final_df, temp_df])
                
        # Need to handle errors otherwise code stops. This is not best practice
        # to simply skip over erro
        except OSError:
            #print('OSError')
            error_report.append([names[i], ':  OSError'])
            pass
        except RuntimeWarning:
            #print('RuntimeWarning')
            error_report.append([names[i], ':  RuntimeWarning'])
            pass
        
        except ValueError:

            error_report.append([names[i], ':  ValueError'])
            pass

        '''
        except KeyError:
            #print('KeyError')
            error_report.append([names[i], ':  HTTP Error 401: UNAUTHORIZED'])
            pass

        # Some devices tested have different error instead of RuntimeWarning
        except sp.optimize._optimize.OptimizeWarning:
            #print('OptimizeWarning')
            error_report.append([names[i], ':  OptimizeWarning'])
            pass 
        '''

        
    # Printing error report
    print('-----------------------------------')
    print('Error report: \n')
    for i in error_report:
        print(i[0], i[1])
    print('-----------------------------------')
    print('Number of networks processed: ', len(nets_processed))
    print('-----------------------------------')
    print('Number of times Tim was better: ', sum(i > j for i, j in zip(r2_tims, r2_HO)))
    print('-----------------------------------')
    print('Number of times HO was better: ', sum(i < j for i, j in zip(r2_tims, r2_HO)))
    print('-----------------------------------')
    print('Number MSE Tim better: ', sum(i < j for i, j in zip(mse_tims, mse_HO)))
    print('-----------------------------------')
    print('Number MSE HO better: ', sum(i > j for i, j in zip(mse_tims, mse_HO)))
    print('-----------------------------------')
    print(r_high_better)
    print('-----------------------------------')
    print(mse_high_better)
    return r2_tims, r2_HO, mse_tims, mse_HO, nets_processed

if __name__ == "__main__":

    # Load in unipartite and run for each real networks
    # Need to get column names for each network from the dataframe
    # Need to do after running get_networks.py
    Unipartite_df = pd.read_pickle('Data/unipartite.pkl')
    upper_node_limit = 23000 # takes too long to run for large networks
    
    unipartite_df = filter_num_verticies(Unipartite_df, upper_node_limit)
    uni_network_names = unipartite_df.columns.values.tolist()

    # Generate file system in /Output with separate folders for each network group
    # Create folder for each network group (if group) and second folder for each network
    MakeFolders(uni_network_names,'RealUniNets')
    # Run analysis on each network
    r2_tim, r2_HO, mse_tims, mse_HO, nets_processed = run_real(uni_network_names)

    # Save dataframe to pickle and HTML
    #df.to_pickle('Output/RealUniNets/RealUniNets_K2_HO.pkl')
    #save_name_html = 'RealUnipartiteNets_2ndDeg_results_HO'
    #write_html(df, save_name_html)

    end = time.time()
    print('Time taken: ', end-start)

    # Save R2 and MSE values
    r2_df = pd.DataFrame({'Network': nets_processed, 'R2 TIM': r2_tim, 'R2 HO': r2_HO,
                            'MSE TIM': mse_tims, 'MSE HO': mse_HO})
    r2_df.to_pickle('Output/RealUniNets/RealUniNets_K2_HO_R2_MSE.pkl')

    # plot r2_tim vs r2_HO

    plt.figure(figsize=(5,4))
    plt.plot(r2_tim, r2_HO, 'ro')
    plt.xlabel(r'$r^{2}$'+ ' For Lower Order')
    plt.ylabel(r'$r^{2}$'+ ' For Higher Order')
    # plot y=x between (0,0) and (1,1)
    plt.plot([0, 1], [0, 1], 'k-', lw=2)

    plt.savefig('Output/RealUniNets/r2_tim_vs_r2_HO.svg', dpi=900)
    plt.show()

    # plot mse_tim vs mse_HO

    # plt.figure()
    # plt.plot(mse_tims, mse_HO, 'ro')
    # plt.xlabel(r'$MSE for Lower Order$')
    # plt.ylabel(r'$MSE For Higher Order$')
    # plt.show()

    
