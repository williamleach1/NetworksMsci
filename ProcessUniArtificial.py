from ProcessBase import *
import os
from Plotter import *
# Run for BA, ER and Config. return as dataframe

def run(gen_func, ns, av_deg, name):
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
    
    final_df = pd.DataFrame(columns=["N","1/ln(z)", "1/ln(z) err", "Beta", 
                                "Beta err", "rchi", "pearson r","pearson p-val",
                                "spearmans r","spearmans p-val"])
    plots = Plotter(name)
    
    for n in ns:
        g = gen_func(n, av_deg)
        k, c, a, a_err, b, b_err, rchi, r, rp, rs, rsp, statistics_dict = process(g,func, to_print=False)
        ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
        plots.add_plot(ks,inv_c_mean,errs,label='N = '+ str(n),fitline=True,function=func,popt=[a,b])
        temp_df = pd.DataFrame({"N": n, "1/ln(z)": a, "1/ln(z) err": a_err, "Beta": b, 
                        "Beta err": b_err, "rchi": rchi, "pearson r": r,
                         "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp}, index=[n])
        final_df = pd.concat([final_df, temp_df])
    save_name = 'Output/ArtificialUniNets/' + name + '/K_Inv_C.png'
    plots.plot(legend=True,save=True,savename=save_name)
    return final_df

ns = [1000,2000,4000,8000,16000]
av_degree = 10
names = ['BA','ER']#,'Config']
MakeFolders(names, 'ArtificialUniNets')

df_BA = run(BA, ns, av_degree, 'BA')
print('BA done')
print(df_BA)
df_ER = run(ER, ns, av_degree, 'ER')
print('ER done')
print(df_ER)

html_BA = df_BA.to_html()
html_ER = df_ER.to_html()
# write html to file
text_file = open("Output/index_BA.html", "w")
text_file.write(html_BA)
text_file.close()
text_file = open("Output/index_ER.html", "w")
text_file.write(html_ER)
text_file.close()
