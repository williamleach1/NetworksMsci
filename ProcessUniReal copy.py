from ProcessBase import *
import warnings
import scipy as sp
warnings.filterwarnings("error")
import graph_tool.all as gt
from graph_tool import correlations, generation
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (3.5, 2.8),
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.8,
    'figure.subplot.bottom': 0.13,
    'figure.subplot.top': 0.98,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 6,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.5,
    'lines.markersize': 2,
})

import math
import matplotlib.pyplot as plt

def run_real(names):
    error_report = []
    num = len(names)

    num_figures = math.ceil(num / 5)  # Calculate the number of figures needed
    num_pages = math.ceil(num_figures / 8)  # Calculate the number of pages needed

    pbar = tqdm((range(num)))
    current_figure = 0
    current_page = 0
    j = 0
    for page in range(num_pages):
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8.27, 10.69))  # A4 size in inches
        fig.tight_layout(pad=4.0)  # Add padding between subplots

        for row in range(4):
            for col in range(2):
                if current_figure < num_figures:
                    min_k = 10
                    max_k = 10
                    for i in range(5):#min(5, num - current_figure * 5)):
                        if j >= num:
                            break
                        while True:
                            #pbar.set_postfix({'Network ': names[j]})
                            # Your existing code goes here, but use the current axes for plotting
                            # (e.g., replace plt.plot with axes[row, col].plot)
                            try:
                                if j >= num:
                                    break
                                g = load_graph(names[j])
                                # Make the name by splitting at '/' if it existts and replace with '_'
                                # This is to make the name of the file the same as the name of the graph
                                # if the graph is loaded from a file
                                if '/' in names[j]:
                                    name = names[j].split('/')
                                    name = name[0]+'_'+name[1]
                                else:
                                    name = names[j]
                                # Now process the graph
                                k, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k= process(g, 2, Real = True, Name = name )
                                inv_c = 1/c
                                a = popt[0]
                                b = popt[1]
                                ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
                                if len(ks) > 100:
                                    mew = 0.25
                                    alpha = 0.7
                                else:
                                    mew = 0.5
                                    alpha = 1
                                


                                scaled_ks = (np.log(ks)-np.log(min(ks)))/(np.log(max(ks))-np.log(min(ks)))
                                folder = 'Output/RealUniNets/' + names[j] + '/'
                                # if max(ks) > max_k:
                                #     max_k = max(ks)
                                # if min(ks) < min_k:
                                #     min_k = min(ks)
                                # Now for collapse plot
                                inv_c_pred = Tim(ks,a,b)
                                y = inv_c_mean/inv_c_pred
                                y_err = errs/inv_c_pred
                                # Shade +/- 0.05
                                
                                axes[row,col].errorbar(scaled_ks, y, yerr=y_err, fmt='none' ,markersize = 0.5,capsize=1,color='grey', elinewidth=0.25, capthick=0.25, alpha = alpha)
                                axes[row,col].plot(scaled_ks, y, 'o',mfc='none', markersize = 2,mew=mew, label = names[j], alpha = alpha)
                                j += 1
                                break 
                            # Need to handle errors otherwise code stops. This is not best practice
                            # to simply skip over erro
                            except OSError:
                                error_report.append([names[i], ':  OSError'])
                                j += 1
                                # repeat the loop
                                continue
                            except KeyError:
                                error_report.append([names[i], ':  HTTP Error 401: UNAUTHORIZED'])
                                j += 1
                                # repeat the loop
                                continue
                            except RuntimeWarning:
                                error_report.append([names[i], ':  RuntimeWarning'])
                                j += 1
                                # repeat the loop
                            # Some devices tested have different error instead of RuntimeWarning
                            except sp.optimize._optimize.OptimizeWarning:
                                error_report.append([names[i], ':  OptimizeWarning'])
                                j += 1
                                continue
                            except ValueError:
                                error_report.append([names[i], ':  ValueError'])
                                j += 1
                                continue
                            # Add your data series to the current subplot using axes[row, col]
                            # e.g., axes[row, col].plot(x, y, label='Data Series {}'.format(i + 1))
                    fill_ks = np.linspace(0, 1, 100)
                    axes[row, col].fill_between(fill_ks, 1-0.05, 1+0.05, color='yellow', alpha=0.2, edgecolor = 'none')#, label=r'$\pm 5\%$')
                    axes[row, col].fill_between(fill_ks, 1+0.05, 1+0.1, color='grey', alpha=0.2, edgecolor = 'none')#, label=r'$\pm 10\%$')
                    axes[row, col].fill_between(fill_ks, 1-0.1, 1-0.05, color='grey', alpha=0.2, edgecolor = 'none')#, label=r'$\pm 10\%$')
                    axes[row, col].set_xlabel(r"$\left[\ln(k)-\ln(k_{min}\right]/\left[\ln(k_{max})-\ln(k_{min})\right]$")
                    y_margin = 0.0  # Adjust this value to change the spacing at the top of the y-axis
                    y_min, y_max = axes[row,col].get_ylim()

                    # Set the new y-axis limits
                    axes[row,col].set_ylim(y_min, y_max+y_margin)
                    if col == 0:
                        axes[row, col].set_ylabel(r"$\dfrac{\hat{c}}{c}$", rotation=0, labelpad=10)
                        axes[row, col].legend(fontsize=6,loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1), fancybox=True,edgecolor='black', facecolor=(1,1,1,1),
                                              handlelength=1, handletextpad=0.5, borderaxespad=0, columnspacing=1, markerscale=2)
                    if col == 1:
                        right_yaxis = axes[row, col].twinx()
                        right_yaxis.set_ylabel(r"$\dfrac{\hat{c}}{c}$", rotation=0, labelpad=10)
                        axes[row, col].legend(fontsize=6,loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1), fancybox=True,edgecolor='black', facecolor=(1,1,1,1),
                                              handlelength=1, handletextpad=0.5, borderaxespad=0, columnspacing=1, markerscale=2)

                        # Set the limits of the right y-axis to match the limits of the left y-axis
                        right_yaxis.set_ylim(axes[row, col].get_ylim())
                        axes[row, col].set_yticks([])
                    num_lab = [[0,1],[2,3],[4,5],[6,7],[8,9]]
                    axes[row, col].set_title('({})'.format(chr(ord('a') + num_lab[row][col])), loc='left', y=-0.3, fontsize=10)
                    current_figure += 1
                else:
                    fig.delaxes(axes[row, col])  # Delete any unused subplot

        # Add a global title to the page
        plt.subplots_adjust(top=0.96, bottom = 0.05, right = 0.92, left =0.08)  # Adjust the top space to accommodate the global title

        fig.subplots_adjust(hspace = 0.5, wspace = 0.1)
        #fig.tight_layout()
        # Save the page as a separate file
        plt.savefig('Output/RealUniNets/K2_Page_{}.png'.format(current_page + 1), dpi=900)
        plt.close()

        current_page += 1
    pbar.close()

# Load in unipartite and run for each real networks
# Need to get column names for each network from the dataframe
# Need to do after running get_networks.py
Unipartite_df = pd.read_pickle('Output/RealUniNets/RealUniNets_K2.pkl')

# Sort by median_degree and rchi
#Unipartite_df = Unipartite_df.sort_values(by=["clustering"], ascending=True)

print(Unipartite_df['rchi_second'] )
# Remove if rchi < 1
Unipartite_df = Unipartite_df[Unipartite_df['rchi_second'] > 1]
# sort by alphabetic index
Unipartite_df = Unipartite_df.sort_index()

uni_network_names = Unipartite_df.index.values.tolist()
# print number
print(len(uni_network_names))

# Run analysis on each network
run_real(uni_network_names)


