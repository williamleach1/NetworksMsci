from ProcessBase import *
import matplotlib.cm as cm


params =    {'font.size' : 16,
            'axes.labelsize':16*2,
            'axes.labelpad': 20,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'axes.titlesize': 16,
            'figure.titlesize': 16,
            'figure.figsize': (12, 9),}

plt.rcParams.update(params)

def package(g):
    g = clean_graph(g)
    k, c, popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k = process(g,1,Real=False)
    ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
    unagg = [k, 1/c]
    agg = [ks, inv_c_mean, errs]
    agg_pred = [ks, Tim(ks, *popt)]
    return unagg, agg, agg_pred, rchi

def package_HO(g):
    g = clean_graph(g)
    k, c, popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k = process(g,3, Real=False)
    ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
    unagg = [k, 1/c]
    agg = [ks, inv_c_mean, errs]
    agg_pred = [ks, HO_Tim(ks, *popt)]
    return unagg, agg, agg_pred

def get_density(g):
    num_verticies = len(g.get_vertices())
    num_edges = len(g.get_edges())
    # find densiy of network which is number of edges/number of possible edges
    density = num_edges/(num_verticies*(num_verticies-1)/2)
    return density

def get_clustering(g):
    (C, var_c) = gt.global_clustering(g)
    return C

names = ['BA','ER']
MakeFolders(names, 'ArtificialUniNets')

# Plot one for BA - various mean k for one N
# Also do a collapse with multiple displayed on one plot
'''
folder = "Output/ArtificialUniNets/BA/"
figA, axsA = plt.subplots(1,1)

figs, axs = plt.subplots(1, 2, figsize=(15, 10))

N = 2000
av_degs = [10,20,40,80,160,320]

for i in range(len(av_degs)):
    g = BA(N, av_degs[i])
    unagg, agg, agg_pred = package(g)
    #unagg_HO, agg_HO, agg_pred_HO = package_HO(g)

    plt.figure()
    index = np.arange(len(unagg[0])).astype(float)
    # Color by index

    plt.scatter(unagg[0], unagg[1],c=index,cmap = cm.plasma ,
            label = r'$\langle k \rangle , N = $' + str(av_degs[i])+', '+str(N)) 
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\frac{1}{c}$", rotation=0)
    plt.xscale("log")
    plt.legend()
    plt.savefig(folder + 'BA_Unagg_'+str(N)+'_'+str(av_degs[i])+ '.svg', dpi = 900)
    plt.close()

    k, c = unagg
    ks, inv_c_mean, errs = agg
    ks, inv_c_pred = agg_pred
    #ks_HO, inv_c_pred_HO = agg_pred_HO
    axsA.errorbar(ks, inv_c_mean, yerr=errs, fmt='.' ,markersize = 5,capsize=2,color='black')
    axsA.plot(ks, inv_c_mean,'o', label=r'$\langle k \rangle = $' + str(av_degs[i]))
    axsA.plot(ks, inv_c_pred,'k--')
    #axsA.plot(ks_HO, inv_c_pred_HO,'b--')

    y = inv_c_mean/inv_c_pred
    y_err = errs/inv_c_pred
    axs[0].errorbar(ks, y, yerr=y_err, fmt='o' ,markersize = 2,capsize=2,color='black')
    axs[0].plot(ks, y,'o', markersize = 5, label = r'$\langle k \rangle = $' + str(av_degs[i]))
    

    c_mean = 1/inv_c_mean
    c_pred = 1/inv_c_pred
    axs[1].plot(c_pred, c_mean, 'o', label = r'$\langle k \rangle = $' + str(av_degs[i]))

axsA.legend()
axsA.set_xlabel(r"$k$")
axsA.set_ylabel(r"$\frac{1}{c}$", rotation=0)
axsA.set_xscale("log")

min_k, max_k = axs[0].get_xlim()

fill_x = np.linspace(min_k*0.95, max_k*1.05, 100)

axs[0].fill_between(fill_x, 1-0.05, 1+0.05, color='grey', alpha=0.2, label=r'$\pm 5\%$')
axs[0].set_xlabel(r"$k$")
axs[0].set_ylabel(r"$\frac{\hat{c}}{c}$", rotation=0)
axs[0].legend()
axs[0].set_xscale("log")

min_y, max_y = axs[1].get_ylim()
min_x, max_x = axs[1].get_xlim()

c_pred = np.linspace(min_x*0.95, max_x*1.05, 100)

axs[1].plot(c_pred, c_pred, 'k--', label='Expected')
axs[1].fill_between(c_pred, c_pred*(1-0.05), c_pred*(1+0.05), color='grey', alpha=0.2, label=r'$\pm 5\%$')
axs[1].set_xlabel(r"$\hat{c}$", rotation=0)
axs[1].set_ylabel(r"$c$", rotation=0)
axs[1].legend()
figA.savefig(folder + 'Av_Deg_BA_N20000.svg', dpi = 900)
figs.savefig(folder+ 'Av_Deg_BA_collapse_N20000.svg', dpi= 900)

folder = "Output/ArtificialUniNets/BA/"
figA, axsA = plt.subplots(1,1)

figs, axs = plt.subplots(1, 2, figsize=(15, 10))

Ns = [1000, 2000, 4000, 8000]
av_deg = 10

for i in range(len(Ns)):
    g = BA(Ns[i], av_deg)
    unagg, agg, agg_pred = package(g)
    #unagg_HO, agg_HO, agg_pred_HO = package_HO(g)

    plt.figure()
    plt.plot(unagg[0], unagg[1], 'o', label = r'$\langle k \rangle , N = $' + str(Ns[i])+', '+str(av_deg)) 
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\frac{1}{c}$", rotation=0)
    plt.xscale("log")
    plt.legend()
    plt.savefig(folder + 'BA_Unagg_'+str(av_deg)+'_'+str(Ns[i])+ '.svg', dpi = 900)
    plt.close()

    k, c = unagg
    ks, inv_c_mean, errs = agg
    ks, inv_c_pred = agg_pred
    #ks_HO, inv_c_pred_HO = agg_pred_HO
    axsA.errorbar(ks, inv_c_mean, yerr=errs, fmt='.' ,markersize = 5,capsize=2,color='black')
    axsA.plot(ks, inv_c_mean,'o', label=r'$N = $' + str(Ns[i]))
    axsA.plot(ks, inv_c_pred,'k--')
    #axsA.plot(ks_HO, inv_c_pred_HO,'b--')

    y = inv_c_mean/inv_c_pred
    y_err = errs/inv_c_pred
    axs[0].errorbar(ks, y, yerr=y_err, fmt='o' ,markersize = 2,capsize=2,color='black')
    axs[0].plot(ks, y,'o', markersize = 5, label = r'$N = $' + str(Ns[i]))
    
    c_mean = 1/inv_c_mean
    c_pred = 1/inv_c_pred
    axs[1].plot(c_pred, c_mean, 'o', label = r'$N = $' + str(Ns[i]))

axsA.legend()
axsA.set_xlabel(r"$k$")
axsA.set_ylabel(r"$\frac{1}{c}$", rotation=0)
axsA.set_xscale("log")

min_k, max_k = axs[0].get_xlim()

fill_x = np.linspace(min_k*0.95, max_k*1.05, 100)

axs[0].fill_between(fill_x, 1-0.05, 1+0.05, color='grey', alpha=0.2, label=r'$\pm 5\%$')
axs[0].set_xlabel(r"$k$")
axs[0].set_ylabel(r"$\frac{\hat{c}}{c}$", rotation=0)
axs[0].legend()
axs[0].set_xscale("log")

min_y, max_y = axs[1].get_ylim()
min_x, max_x = axs[1].get_xlim()

c_pred = np.linspace(min_x*0.95, max_x*1.05, 100)

axs[1].plot(c_pred, c_pred, 'k--', label='Expected')
axs[1].fill_between(c_pred, c_pred*(1-0.05), c_pred*(1+0.05), color='grey', alpha=0.2, label=r'$\pm 5\%$')
axs[1].set_xlabel(r"$\hat{c}$", rotation=0)
axs[1].set_ylabel(r"$c$", rotation=0)
axs[1].legend()
figA.savefig(folder + 'Ns_BA_k10.svg', dpi = 900)
figs.savefig(folder+ 'Ns_BA_collapse_k10.svg', dpi= 900)

# Plot one for ER - various mean k for one N
# Also do a collapse with multiple displayed on one plot

folder = "Output/ArtificialUniNets/ER/"
figA, axsA = plt.subplots(1,1)

figs, axs = plt.subplots(1, 2, figsize=(15, 10))

N = 2000
av_degs = [10,20,40,80]

for i in range(len(av_degs)):
    g = ER(N, av_degs[i])
    unagg, agg, agg_pred = package(g)
    #unagg_HO, agg_HO, agg_pred_HO = package_HO(g)

    plt.figure()
    plt.plot(unagg[0], unagg[1], 'o', label = r'$\langle k \rangle , N = $' + str(av_degs[i])+', '+str(N)) 
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\frac{1}{c}$", rotation=0)
    plt.xscale("log")
    plt.legend()
    plt.savefig(folder + 'ER_Unagg_'+str(N)+'_'+str(av_degs[i])+ '.svg', dpi = 900)
    plt.close()

    k, c = unagg
    ks, inv_c_mean, errs = agg
    ks, inv_c_pred = agg_pred
    #ks_HO, inv_c_pred_HO = agg_pred_HO
    axsA.errorbar(ks, inv_c_mean, yerr=errs, fmt='.' ,markersize = 5,capsize=2,color='black')
    axsA.plot(ks, inv_c_mean,'o', label=r'$\langle k \rangle = $' + str(av_degs[i]))
    axsA.plot(ks, inv_c_pred,'k--')
    #axsA.plot(ks_HO, inv_c_pred_HO,'b--')

    y = inv_c_mean/inv_c_pred
    y_err = errs/inv_c_pred
    axs[0].errorbar(ks, y, yerr=y_err, fmt='o' ,markersize = 2,capsize=2,color='black')
    axs[0].plot(ks, y,'o', markersize = 5, label = r'$\langle k \rangle = $' + str(av_degs[i]))
    
    c_mean = 1/inv_c_mean
    c_pred = 1/inv_c_pred
    axs[1].plot(c_pred, c_mean, 'o', label = r'$\langle k \rangle = $' + str(av_degs[i]))

axsA.legend()
axsA.set_xlabel(r"$k$")
axsA.set_ylabel(r"$\frac{1}{c}$", rotation=0)
axsA.set_xscale("log")

min_k, max_k = axs[0].get_xlim()

fill_x = np.linspace(min_k*0.95, max_k*1.05, 100)

axs[0].fill_between(fill_x, 1-0.05, 1+0.05, color='grey', alpha=0.2, label=r'$\pm 5\%$')
axs[0].set_xlabel(r"$k$")
axs[0].set_ylabel(r"$\frac{\hat{c}}{c}$", rotation=0)
axs[0].legend()
axs[0].set_xscale("log")

min_y, max_y = axs[1].get_ylim()
min_x, max_x = axs[1].get_xlim()

c_pred = np.linspace(min_x*0.95, max_x*1.05, 100)

axs[1].plot(c_pred, c_pred, 'k--', label='Expected')
axs[1].fill_between(c_pred, c_pred*(1-0.05), c_pred*(1+0.05), color='grey', alpha=0.2, label=r'$\pm 5\%$')
axs[1].set_xlabel(r"$\hat{c}$", rotation=0)
axs[1].set_ylabel(r"$c$", rotation=0)
axs[1].legend()
figA.savefig(folder + 'Av_Deg_ER_N20000.svg', dpi = 900)
figs.savefig(folder+ 'Av_Deg_ER_collapse_N20000.svg', dpi= 900)
'''
folder = "Output/ArtificialUniNets/ER/"
figA, axsA = plt.subplots(1,1)

figs, axs = plt.subplots(1, 2, figsize=(15, 10))

Ns = [10000, 20000, 40000, 80000, 160000]
av_deg = 10

for i in range(len(Ns)):
    g = ER(Ns[i], av_deg)
    unagg, agg, agg_pred,rchi = package(g)
    #unagg_HO, agg_HO, agg_pred_HO = package_HO(g)

    plt.figure()
    plt.plot(unagg[0], unagg[1], 'o', label = r'$\langle k \rangle , N = $' + str(Ns[i])+', '+str(av_deg)) 
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\dfrac{1}{c}$", rotation=0)
    plt.xscale("log")
    plt.legend()
    plt.savefig(folder + 'ER_Unagg_'+str(av_deg)+'_'+str(Ns[i])+ '.svg', dpi = 900)
    plt.close()

    k, c = unagg
    ks, inv_c_mean, errs = agg
    ks, inv_c_pred = agg_pred
    #ks_HO, inv_c_pred_HO = agg_pred_HO
    rchi_latex = r'$\chi^{2}_{r}$'

    axsA.errorbar(ks, inv_c_mean, yerr=errs, fmt='.' ,markersize = 5,capsize=5,color='black')
    axsA.plot(ks, inv_c_mean,'o', label=r'$N = $' + str(Ns[i])+', '+rchi_latex+' = '+str(round(rchi,3)))
    axsA.plot(ks, inv_c_pred,'k--')
    #axsA.plot(ks_HO, inv_c_pred_HO,'b--')

    y = inv_c_mean/inv_c_pred
    y_err = errs/inv_c_pred
    axs[0].errorbar(ks, y, yerr=y_err, fmt='o' ,markersize = 2,capsize=2,color='black')
    axs[0].plot(ks, y,'o', markersize = 5, label = r'$N = $' + str(Ns[i]))
    
    c_mean = 1/inv_c_mean
    c_pred = 1/inv_c_pred
    axs[1].plot(c_pred, c_mean, 'o', label = r'$N = $' + str(Ns[i]))

axsA.legend(fontsize = 20)
axsA.set_xlabel(r"$k$", rotation=0, labelpad=20, fontsize=25)
axsA.set_ylabel(r"$\dfrac{1}{c}$", rotation=0, labelpad=20, fontsize=25)
axsA.set_xscale("log")

min_k, max_k = axs[0].get_xlim()

fill_x = np.linspace(min_k*0.95, max_k*1.05, 100)

axs[0].fill_between(fill_x, 1-0.05, 1+0.05, color='grey', alpha=0.2, label=r'$\pm 5\%$')
axs[0].set_xlabel(r"$k$")
axs[0].set_ylabel(r"$\frac{\hat{c}}{c}$", rotation=0)
axs[0].legend()
axs[0].set_xscale("log")

min_y, max_y = axs[1].get_ylim()
min_x, max_x = axs[1].get_xlim()

c_pred = np.linspace(min_x*0.95, max_x*1.05, 100)

axs[1].plot(c_pred, c_pred, 'k--', label='Expected')
axs[1].fill_between(c_pred, c_pred*(1-0.05), c_pred*(1+0.05), color='grey', alpha=0.2, label=r'$\pm 5\%$')
axs[1].set_xlabel(r"$\hat{c}$", rotation=0)
axs[1].set_ylabel(r"$c$", rotation=0)
axs[1].legend()
figA.savefig(folder + 'Ns_ER_k10.svg', dpi = 900)
figs.savefig(folder+ 'Ns_ER_collapse_k10.svg', dpi= 900)



# Plot density vs rchi for BA and ER



# Plot clustering vs rchi for BA and ER



# Plot assortativity vs rchi for BA and ER



# Contour plot of BA and ER reduced chi squared for various N and mean k



