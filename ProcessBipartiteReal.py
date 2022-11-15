from ProcessBase import *
import os
from Plotter import *
# Run for BA, ER and Config. return as dataframe

def run(gen_func, ns, av_deg, name,to_html=False,to_print=False):
    """Perform all analysis on graph
    Parameters  
    ----------                  
    gen_func : function
        Function to generate graph
    ns : array
        Array of number of nodes
    av_deg : int
        Average degree
    Returns
    -------     
    df : dataframe
        Dataframe containing results"""
    dfs = pd.DataFrame(columns=["Mean k","N","1/ln(z)", "1/ln(z) err", "Beta", 
                            "Beta err", "rchi", "pearson r","pearson p-val",
                            "spearmans r","spearmans p-val"])
    i =0
    for av_degree in av_deg:
        final_df = pd.DataFrame(columns=["Mean k","N","1/ln(z)", "1/ln(z) err", "Beta", 
                            "Beta err", "rchi", "pearson r","pearson p-val",
                            "spearmans r","spearmans p-val"])
        plots = Plotter(name)
        for n in ns:
            g = gen_func(n, av_degree)
            k, c, a, a_err, b, b_err, rchi, r, rp, rs, rsp, statistics_dict, mean_k = process(g,func, 
                                                                                                to_print=False)
            ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
            plots.add_plot(ks,inv_c_mean,errs,label='N = '+ str(n),fitline=True,function=func,popt=[a,b])
            temp_df = pd.DataFrame({"Mean k": mean_k,"N": n, "1/ln(z)": a, "1/ln(z) err": a_err, "Beta": b, 
                            "Beta err": b_err, "rchi": rchi, "pearson r": r,
                            "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp}, index=[i])
            final_df = pd.concat([final_df, temp_df])
            i+=1
        save_name = 'Output/ArtificialUniNets/' + name + '/K_Inv_C_'+str(av_degree)+'.png'
        plots.plot(legend=True,save=True,savename=save_name)
        dfs = pd.concat([dfs, final_df])
    if to_html:
        save_name_html =  name+'_results'
        write_html(dfs, save_name_html)
    if to_print:
        print(name+ ' done')
        print(dfs)
    return dfs

ns = [1000,2000,4000]
av_degree = [10,20,40]
names = ['BA','ER']#,'Config']
MakeFolders(names, 'ArtificialUniNets')

dfs_BA = run(BA, ns, av_degree, 'BA',to_html=True, to_print=True)
dfs_ER = run(ER, ns, av_degree, 'ER',to_html=True, to_print=True)