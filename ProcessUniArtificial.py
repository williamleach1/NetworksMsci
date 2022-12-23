import os

# Run for BA, ER and Config. return as dataframe
import seaborn as sns
from matplotlib import cm

from Plotter import *
from ProcessBase import *

from datetime import datetime
import csv

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
    # prepare datarame of reduced chi squared. index is number of nodes, column is mean k
    rchi_df = pd.DataFrame(index=ns,columns=av_deg)
    i =0
    for av_degree in av_deg:
        final_df = pd.DataFrame(columns=["Mean k","N","1/ln(z)", "1/ln(z) err", "Beta", 
                            "Beta err", "rchi", "pearson r","pearson p-val",
                            "spearmans r","spearmans p-val"])
        plots = Plotter(name)
        for n in ns:
            g = gen_func(n, av_degree)
            g = clean_graph(g)
            k, c, popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k = process(g,to_print=False)
            a = popt[0]
            b = popt[1]
            a_err = np.sqrt(pcov[0][0])
            b_err = np.sqrt(pcov[1][1])
            ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
            plots.add_plot(ks,inv_c_mean,errs,label='N = '+ str(n),fitline=True,function=Tim,popt=[a,b])
            temp_df = pd.DataFrame({"Mean k": mean_k,"N": n, "1/ln(z)": a, "1/ln(z) err": a_err, "Beta": b, 
                            "Beta err": b_err, "rchi": rchi, "pearson r": r,
                            "pearson p-val": rp, "spearmans r": rs, "spearmans p-val": rsp}, index=[i])
            final_df = pd.concat([final_df, temp_df])
            rchi_df.loc[n,av_degree] = rchi
            i+=1

            os.makedirs('Output/RchiUniArtificial', exist_ok=True)
            # save reduced chi squared to csv file for each graph type
            if name == 'BA':
                row = [datetime.now(),n,av_degree,rchi]
                with open('Output/RchiUniArtificial/rchisBA.csv','a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()
            elif name == 'ER':
                row = [datetime.now(),n,av_degree,rchi]
                with open('Output/RchiUniArtificial/rchisER.csv','a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()

        save_name = 'Output/ArtificialUniNets/' + name + '/K_Inv_C_'+str(av_degree)+'.png'
        plots.plot(legend=True,save=True,savename=save_name)
        dfs = pd.concat([dfs, final_df])
    if to_html:
        save_name_html =  name+'_results'
        write_html(dfs, save_name_html)
    if to_print:
        print(name+ ' done')
        print(dfs)
    return dfs, rchi_df




# ns = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,
#         12000,13000,14000,15000,16000,17000,18000,19000,20000]
# av_degree = [6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,
#              44,46,48,50,52,54,56,58,60]

ns = [1000,4000,7000]#,10000,13000,16000,19000,22000,25000]
av_degree = [6,10,14]#,18,22,26,30,34,38,42,46,50,54,58,62,66,70]
names = ['BA','ER']#,'Config']
MakeFolders(names, 'ArtificialUniNets')
Zs_BA=[]
Zs_ER=[]
repeats = 1

for i in range(repeats):
    dfs_BA, rchi_df = run(BA, ns, av_degree, 'BA',to_html=True, to_print=True)
    dfs_ER, rchi_df2 = run(ER, ns, av_degree, 'ER',to_html=True, to_print=True)
    #make contour plot of reduced chi squared
    Zs_BA.append(rchi_df.to_numpy())
    Zs_ER.append(rchi_df2.to_numpy())
Z_BA = np.mean(Zs_BA,axis=0)
x = np.array(av_degree)
y = np.array(ns)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z_BA, cmap=cm.coolwarm)
fig.colorbar(surface, shrink=0.5, aspect=5)
ax.set_xlabel('Mean k')
ax.set_ylabel('N')
ax.set_zlabel('Reduced Chi Squared')
plt.show()

Z_ER = np.mean(Zs_ER,axis=0)
x = np.array(av_degree)
y = np.array(ns)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z_ER, cmap=cm.coolwarm)
fig.colorbar(surface, shrink=0.5, aspect=5)
ax.set_xlabel('Mean k')
ax.set_ylabel('N')
ax.set_zlabel('Reduced Chi Squared')
plt.show()





