from ProcessBase import *
from IPython.display import HTML
import warnings
import scipy as sp
from Plotter import *
warnings.filterwarnings("error")

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
    final_df = pd.DataFrame(columns=["N","1/ln(z)", "1/ln(z) err", "Beta",
                                "Beta err", "rchi", "pearson r","pearson p-val",
                                "spearmans r","spearmans p-val"])
    error_report = []
    num = len(names)
    pbar = tqdm((range(num)))
    for i in pbar:
        pbar.set_postfix({'Network ': names[i]})
        try:
            g = load_graph(names[i])
            k, c, a, a_err, b, b_err, rchi, r, rp, rs, rsp, statistics_dict= process(g,func, to_print=False)
            plots = Plotter(names[i])
            ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
            plots.add_plot(ks,inv_c_mean,yerr=errs,fitline=True,function=func,popt=[a,b])
            save_name = 'Output/RealUniNets/' + names[i] + '/K_Inv_C.png'
            plots.plot(save=True,savename=save_name)
            temp_df = pd.DataFrame({"N": len(g.get_vertices()), "1/ln(z)": a, "1/ln(z) err": a_err, 
                                "Beta": b, "Beta err": b_err, "rchi": rchi, "pearson r": r,
                                "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp}, 
                                index=[names[i]])
            final_df = pd.concat([final_df, temp_df])
        except OSError:
            error_report.append([names[i], ':  OSError'])
            pass
        except KeyError:
            error_report.append([names[i], ':  HTTP Error 401: UNAUTHORIZED'])
            pass
        except RuntimeWarning:
            error_report.append([names[i], ':  RuntimeWarning'])
            pass
    print('-----------------------------------')
    print('Error report: \n')
    for i in error_report:
        print(i[0], i[1])
    print('-----------------------------------')
    return final_df

# Load in unipartite and run for each real networks
# Need to get column names for each network from the dataframe
unipartite_df = pd.read_pickle('Data/unipartite.pkl')

# Filter out num_vertices>2000000
unipartite_df = unipartite_df.transpose()
unipartite_df = unipartite_df.loc[unipartite_df['num_vertices']<50000,]
unipartite_df = unipartite_df.transpose()
uni_network_names = unipartite_df.columns.values.tolist()
print(len(uni_network_names))


# Generate file system in /Output with separate folders for each network group
# Create folder for each network group (if group) and second folder for each network


MakeFolders(uni_network_names,'RealUniNets')


df = run_real(uni_network_names)
print(df)

html = df.to_html()
# write html to file
text_file = open("Output/RealUnipartiteNets.html", "w")
text_file.write(html)
text_file.close()
end = time.time()

print('Time taken: ', end-start)
