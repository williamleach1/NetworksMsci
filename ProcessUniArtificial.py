from ProcessBase import *
import matplotlib.cm as cm

# Do one for padding
plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (3.5, 2.8),
    'figure.subplot.left': 0.15,
    'figure.subplot.right': 0.98,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.top': 0.98,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.5,
    'lines.markersize': 2,
})

# Prepare dataframe to store results
columns = ["Model Name", "N", "mean_k", "z", "z err", "Beta", "Beta_err", "rchi"]

main_df = pd.DataFrame(columns=columns)


def get_density(g):
    num_verticies = len(g.get_vertices())
    num_edges = len(g.get_edges())
    # find densiy of network which is number of edges/number of possible edges
    density = num_edges/(num_verticies*(num_verticies-1)/2)
    return density

def get_clustering(g):
    (C, var_c) = gt.global_clustering(g)
    return C

def package(g, Name, model):
    g = clean_graph(g)
    k, c, popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k = process(g,1,Real=True, Name=Name)
    ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
    unagg = [k, 1/c]
    agg = [ks, inv_c_mean, errs]
    agg_pred = [ks, Tim(ks, *popt)]
    inv_ln_z = popt[0]
    beta = popt[1]
    inv_ln_z_err = np.sqrt(pcov[0][0])
    beta_err = np.sqrt(pcov[1][1])
    inv_ln_z_u = ufloat(inv_ln_z, inv_ln_z_err)
    beta_u = ufloat(beta, beta_err)

    ln_z_u = 1/inv_ln_z_u
    z_u = umath.exp(ln_z_u)
    z = z_u.n
    z_err = z_u.s

    columns = ["Model Name", "N", "mean_k", "z", "z err", "Beta", "Beta_err", "rchi"]

    df = pd.DataFrame([[model, len(g.get_vertices()), mean_k, z, z_err, beta, beta_err, rchi]], columns=columns)

    # round columns (except name and N) to 3 decimal places
    df.iloc[:,2:] = df.iloc[:,2:].round(3)

    return unagg, agg, agg_pred, rchi, df

def package_HO(g):
    g = clean_graph(g)
    k, c, popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k = process(g,3, Real=False)
    ks, inv_c_mean, errs, stds, counts   = unpack_stat_dict(statistics_dict)
    unagg = [k, 1/c]
    agg = [ks, inv_c_mean, errs]
    agg_pred = [ks, HO_Tim(ks, *popt)]
    return unagg, agg, agg_pred, rchi

names = ['BA','ER']
MakeFolders(names, 'ArtificialUniNets')

# Plot one for BA - various mean k for one N
# Also do a collapse with multiple displayed on one plot

folder = "Output/ArtificialUniNets/BA/"
# Separate figures for each subplot
figA, axsA = plt.subplots()

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

N = 10000
av_degs = [10, 20, 40, 80, 160]

for i in range(len(av_degs)):
    g = BA(N, av_degs[i])
    Name = 'BA_' + str(N) + '_' + str(av_degs[i])
    unagg, agg, agg_pred, rchi, df = package(g, Name, 'BA')

    main_df = pd.concat([main_df, df])

    k, c = unagg
    ks, inv_c_mean, errs = agg
    ks, inv_c_pred = agg_pred
    rchi_latex = r'$\chi^{2}_{r}$'
    axsA.errorbar(ks, inv_c_mean, yerr=errs, fmt='none',markersize=0.5, capsize=1, color='grey', elinewidth=0.25, capthick=0.25)
    axsA.plot(ks, inv_c_mean, 'o',mfc='none' ,mew=0.5,markersize=2, alpha=1, label=r'$\langle k \rangle = $' + str(av_degs[i]), )
    axsA.plot(ks, inv_c_pred, 'k--')

    y = inv_c_mean / inv_c_pred
    y_err = errs / inv_c_pred
    ax1.errorbar(ks, y, yerr=y_err, fmt='none',markersize=0.5, capsize=1 , color='grey', elinewidth=0.25, capthick=0.25)
    ax1.plot(ks, y, 'o',mfc='none' ,mew=0.5, label=r'$\langle k \rangle = $' + str(av_degs[i]),markersize=2 , alpha=1)

    c_mean = 1 / inv_c_mean
    c_pred = 1 / inv_c_pred
    ax2.plot(c_pred, c_mean, 'o',mfc='none' ,mew=0.5, label=r'$\langle k \rangle = $' + str(av_degs[i]), markersize=2, alpha=1)

axsA.legend()
axsA.set_xlabel(r"$k$")
axsA.set_ylabel(r"$\dfrac{1}{c}$", rotation=0, labelpad=10)
axsA.set_xscale("log")

min_k, max_k = ax1.get_xlim()
fill_x = np.linspace(min_k * 0.95, max_k * 1.05, 100)

ax1.fill_between(fill_x, 1 - 0.02, 1 + 0.02, color='grey', alpha=0.2, label=r'$\pm 2\%$')
ax1.set_xlabel(r"$k$")
ax1.set_ylabel(r"$\dfrac{\hat{c}}{c}$", rotation=0, labelpad=10)
ax1.legend()
fig1.subplots_adjust(left= 0.2)
ax1.set_xscale("log")

min_y, max_y = ax2.get_ylim()
min_x, max_x = ax2.get_xlim()
c_pred = np.linspace(min_x * 0.95, max_x * 1.05, 100)

ax2.plot(c_pred, c_pred, 'k--', label='Expected')
ax2.fill_between(c_pred, c_pred * (1 - 0.05), c_pred * (1 + 0.05), color='grey', alpha=0.2, label=r'$\pm 5\%$')
ax2.set_xlabel(r"$\hat{c}$", rotation=0)
ax2.set_ylabel(r"$c$", rotation=0, labelpad=10)
ax2.legend()

# Save the figures
figA.savefig(folder + 'Av_Deg_BA_N20000.png', dpi=900)
fig1.savefig(folder + 'Av_Deg_BA_collapse_N20000_subplot1.png', dpi=900)
fig2.savefig(folder + 'Av_Deg_BA_collapse_N20000_subplot2.png', dpi=900)

folder = "Output/ArtificialUniNets/BA/"

# Separate figures for each subplot
figA, axsA = plt.subplots()

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

Ns = [10000, 20000, 40000, 80000, 160000]
av_deg = 10

for i in range(len(Ns)):
    g = BA(Ns[i], av_deg)
    Name = 'BA_' + str(Ns[i]) + '_' + str(av_deg)
    unagg, agg, agg_pred, rchi, df = package(g, Name, 'BA')
    
    main_df = pd.concat([main_df, df])
    k, c = unagg
    ks, inv_c_mean, errs = agg
    ks, inv_c_pred = agg_pred
    rchi_latex = r'$\chi^{2}_{r}$'
    extra = r'$N = $' + str(Ns[i]) +', '+rchi_latex+' = '+str(round(rchi,3))
    axsA.errorbar(ks, inv_c_mean, yerr=errs, fmt='none', markersize=0.5, capsize=1, color='grey', elinewidth=0.25, capthick=0.25)
    axsA.plot(ks, inv_c_mean, 'o',mfc='none' ,mew=0.5, label=r'$N = $' + str(Ns[i]), markersize=2, alpha=1)
    axsA.plot(ks, inv_c_pred, 'k--')

    y = inv_c_mean / inv_c_pred
    y_err = errs / inv_c_pred
    ax1.errorbar(ks, y, yerr=y_err, fmt='none', markersize=0.5, capsize=1, color='grey', elinewidth=0.25, capthick=0.25)
    ax1.plot(ks, y, 'o',mfc='none' ,mew=0.5, markersize=2, label=r'$N = $' + str(Ns[i]))

    c_mean = 1 / inv_c_mean
    c_pred = 1 / inv_c_pred
    ax2.plot(c_pred, c_mean, 'o',mfc='none' ,mew=0.5, label=r'$N = $' + str(Ns[i]), markersize=2, alpha=1)

axsA.legend()
axsA.set_xlabel(r"$k$")
axsA.set_ylabel(r"$\dfrac{1}{c}$", rotation=0, labelpad=10)
axsA.set_xscale("log")

min_k, max_k = ax1.get_xlim()
fill_x = np.linspace(min_k * 0.95, max_k * 1.05, 100)

ax1.fill_between(fill_x, 1 - 0.02, 1 + 0.02, color='grey', alpha=0.2, label=r'$\pm 2\%$')
ax1.set_xlabel(r"$k$")
ax1.set_ylabel(r"$\dfrac{\hat{c}}{c}$", rotation=0, labelpad=10)
ax1.legend()
ax1.set_xscale("log")

min_y, max_y = ax2.get_ylim()
min_x, max_x = ax2.get_xlim()
c_pred = np.linspace(min_x * 0.95, max_x * 1.05, 100)

ax2.plot(c_pred, c_pred, 'k--', label='Expected')
ax2.fill_between(c_pred, c_pred * (1 - 0.05), c_pred * (1 + 0.05), color='grey', alpha=0.2, label=r'$\pm 5\%$')
ax2.set_xlabel(r"$\hat{c}$", rotation=0)
ax2.set_ylabel(r"$c$", rotation=0, labelpad=10)
ax2.legend()

# Save the figures
figA.savefig(folder + 'Ns_BA_k10.png', dpi=900)
fig1.savefig(folder + 'Ns_BA_collapse_k10_subplot1.png', dpi=900)
fig2.savefig(folder + 'Ns_BA_collapse_k10_subplot2.png', dpi=900)

# Plot one for ER - various mean k for one N
# Also do a collapse with multiple displayed on one plot

folder = "Output/ArtificialUniNets/ER/"

figA, axsA = plt.subplots(1,1)

fig1, axs1 = plt.subplots(1,1)
fig2, axs2 = plt.subplots(1,1)

N = 10000
av_degs = [10, 20, 40, 80, 160]

for i in range(len(av_degs)):
    g = ER(N, av_degs[i])
    Name = 'ER_'+str(N)+'_'+str(av_degs[i])
    unagg, agg, agg_pred, rchi, df = package(g,Name, 'ER')
    
    main_df = pd.concat([main_df, df])
    k, c = unagg
    ks, inv_c_mean, errs = agg
    ks, inv_c_pred = agg_pred
    rchi_latex = r'$\chi^{2}_{r}$'
    axsA.errorbar(ks, inv_c_mean, yerr=errs, fmt='none' ,markersize = 0.5,capsize=1,color='grey', elinewidth=0.25, capthick=0.25)
    axsA.plot(ks, inv_c_mean,'o',mfc='none' ,mew=0.5, label=r'$\langle k \rangle = $' + str(av_degs[i]), markersize = 2, alpha=1)
    axsA.plot(ks, inv_c_pred,'k--')

    y = inv_c_mean/inv_c_pred
    y_err = errs/inv_c_pred
    axs1.errorbar(ks, y, yerr=y_err, fmt='none' ,markersize = 0.5,capsize=1,color='grey', elinewidth=0.25, capthick=0.25)
    axs1.plot(ks, y,'o', markersize = 2,mfc='none' ,mew=0.5, label = r'$\langle k \rangle = $' + str(av_degs[i]), alpha=1)

    c_mean = 1/inv_c_mean
    c_pred = 1/inv_c_pred
    axs2.plot(c_pred, c_mean, 'o',mfc='none' ,mew=0.5, label = r'$\langle k \rangle = $' + str(av_degs[i]), markersize = 2, alpha=1)

# Plot settings for figA
axsA.legend()
axsA.set_xlabel(r"$k$")
axsA.set_ylabel(r"$\dfrac{1}{c}$", rotation=0, labelpad=10)
axsA.set_xscale("log")

# Plot settings for fig1
min_k, max_k = axs1.get_xlim()
fill_x = np.linspace(min_k*0.95, max_k*1.05, 100)

axs1.fill_between(fill_x, 1-0.02, 1+0.02, color='grey', alpha=0.2, label=r'$\pm 2\%$')
axs1.set_xlabel(r"$k$")
axs1.set_ylabel(r"$\dfrac{\hat{c}}{c}$", rotation=0, labelpad=10)
axs1.legend()
axs1.set_xscale("log")

# Plot settings for fig2
min_y, max_y = axs2.get_ylim()
min_x, max_x = axs2.get_xlim()

c_pred = np.linspace(min_x*0.95, max_x*1.05, 100)

axs2.plot(c_pred, c_pred, 'k--', label='Expected')
axs2.fill_between(c_pred, c_pred*(1-0.05), c_pred*(1+0.05), color='grey', alpha=0.2, label=r'$\pm 5\%$')
axs2.set_xlabel(r"$\hat{c}$", rotation=0)
axs2.set_ylabel(r"$c$", rotation=0, labelpad=10)
axs2.legend()

# Save figures
figA.savefig(folder + 'Av_Deg_ER_N20000_figA.png', dpi = 900)
fig1.savefig(folder + 'Av_Deg_ER_N20000_fig1.png', dpi = 900)
fig2.savefig(folder + 'Av_Deg_ER_N20000_fig2.png', dpi = 900)

folder = "Output/ArtificialUniNets/ER/"
figA, axsA = plt.subplots(1,1)

fig1, axs1 = plt.subplots(1,1)
fig2, axs2 = plt.subplots(1,1)

Ns = [10000, 20000, 40000, 80000, 160000]
av_deg = 10

for i in range(len(Ns)):
    g = ER(Ns[i], av_deg)
    Name = 'ER_'+str(Ns[i])+'_'+str(av_deg)
    unagg, agg, agg_pred, rchi, df = package(g, Name, 'ER')
    main_df = pd.concat([main_df, df])

    # Building networks, one edge at a time.
    k, c = unagg
    ks, inv_c_mean, errs = agg
    ks, inv_c_pred = agg_pred
    rchi_latex = r'$\chi^{2}_{r}$'
    extra = rchi_latex+' = '+str(round(rchi,3))
    axsA.errorbar(ks, inv_c_mean, yerr=errs, fmt='none' ,markersize = 0.5,capsize=1,color='grey', elinewidth=0.25, capthick=0.25)
    axsA.plot(ks, inv_c_mean,'o',mfc='none' ,mew=0.5, label=r'$N = $' + str(Ns[i]), markersize = 2)
    axsA.plot(ks, inv_c_pred,'k--')

    y = inv_c_mean/inv_c_pred
    y_err = errs/inv_c_pred
    axs1.errorbar(ks, y, yerr=y_err, fmt='none' ,markersize = 0.5,capsize=1,color='grey', elinewidth=0.25, capthick=0.25)
    axs1.plot(ks, y,'o',mfc='none' ,mew=0.5, markersize = 2, label = r'$N = $' + str(Ns[i]))
    
    c_mean = 1/inv_c_mean
    c_pred = 1/inv_c_pred
    axs2.plot(c_pred, c_mean, 'o',mfc='none' ,mew=0.5, label = r'$N = $' + str(Ns[i]), markersize = 2)

axsA.legend()
axsA.set_xlabel(r"$k$", rotation=0)
axsA.set_ylabel(r"$\dfrac{1}{c}$", rotation=0, labelpad=10)
axsA.set_xscale("log")

min_k, max_k = axs1.get_xlim()

fill_x = np.linspace(min_k*0.95, max_k*1.05, 100)

axs1.fill_between(fill_x, 1-0.02, 1+0.02, color='grey', alpha=0.2, label=r'$\pm 2\%$')
axs1.set_xlabel(r"$k$")
axs1.set_ylabel(r"$\dfrac{\hat{c}}{c}$", rotation=0, labelpad=10)
axs1.legend()
axs1.set_xscale("log")

min_y, max_y = axs2.get_ylim()
min_x, max_x = axs2.get_xlim()

c_pred = np.linspace(min_x*0.95, max_x*1.05, 100)

axs2.plot(c_pred, c_pred, 'k--', label='Expected')
axs2.fill_between(c_pred, c_pred*(1-0.05), c_pred*(1+0.05), color='grey', alpha=0.2, label=r'$\pm 5\%$')
axs2.set_xlabel(r"$\hat{c}$", rotation=0)
axs2.set_ylabel(r"$c$", rotation=0, labelpad=10)
axs2.legend()
figA.savefig(folder + 'Ns_ER_k10.png', dpi = 900)
fig1.savefig(folder+ 'Ns_ER_collapse_k10_fig1.png', dpi= 900)
fig2.savefig(folder+ 'Ns_ER_collapse_k10_fig2.png', dpi= 900)

# Plot density vs rchi for BA and ER
folder = "Output/ArtificialUniNets/"

main_df.to_latex(folder + 'Results.tex',
                index=False,
                header=True,
                bold_rows=True,
                float_format="%.3f",
                caption="My DataFrame",
                label="tab:my_dataframe",
                column_format="|c|c|c|")

# Plot clustering vs rchi for BA and ER



# Plot assortativity vs rchi for BA and ER



# Contour plot of BA and ER reduced chi squared for various N and mean k



