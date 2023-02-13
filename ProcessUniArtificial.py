import os
import seaborn as sns
from matplotlib import cm
from ProcessBase import *
from datetime import datetime
import csv


def package(g):
    g = clean_graph(g)
    k, c, popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k = process(g,1)
    ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
    rchi = red_chi_square(k,c,  )
    unagg = [k, c]
    agg = [ks, inv_c_mean, errs]
    agg_pred = [ks, Tim(ks, *popt)]
    return unagg, agg, agg_pred, rchi

def get_density(g):
    num_verticies = len(g.get_vertices())
    num_edges = len(g.get_edges())
    # find densiy of network which is number of edges/number of possible edges
    density = num_edges/(num_verticies*(num_verticies-1)/2)
    return density

def get_clustering(g):
    (C, var_c) = gt.global_clustering(g)
    return C

# Plot one for BA - various mean k for one N
# Also do a collapse with multiple displayed on one plot


plt.figure()

for i in range
plt.errorbar(ks, inv_c_mean, yerr=errs, fmt='.' ,markersize = 5,capsize=2,color='black')
plt.plot(ks, inv_c_mean,'ro', label="Data")
plt.plot(ks, Tim(ks, *popt),'b--', label="Fit to data")

plt.legend()
plt.xlabel(r"$k$")
plt.ylabel(r"$\frac{1}{c}$", rotation=0)
plt.xscale("log")
plt.title(names[i])
plt.savefig(folder + 'inv_c_vs_k_agg.svg', dpi=900)
plt.close()

# Now for collapse plot
inv_c_pred = Tim(ks,a,b)
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
plt.savefig(folder + 'inv_c_vs_k_collapse.svg', dpi=900)
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

plt.savefig(folder + 'c_vs_c_hat.svg', dpi=900)
plt.close()






# Plot one for ER - various mean k for one N
# Also do a collapse with multiple displayed on one plot



# Plot density vs rchi for BA and ER



# Plot clustering vs rchi for BA and ER



# Contour plot of BA and ER reduced chi squared for various N and mean k







'''
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
'''





