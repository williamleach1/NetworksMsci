import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AnalysisBase import *
import seaborn as sns
# Load in analysed dataframes
plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (5, 3.5),
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

df_uni_real = pd.read_pickle('Output/RealUniNets/RealUniNets.pkl')

df_bi_real = pd.read_pickle('Output/RealBipartiteNets/RealBipartiteNets.pkl')

Unipartite_df = pd.read_pickle('Data/unipartite.pkl')

# transpose the dataframes so that the index is the network name
Unipartite_df = Unipartite_df.transpose()

# load completed dataframes

processed_networks = get_intersection_index(df_uni_real, Unipartite_df)

# Take the subset of the unipartite dataframe that has not processed
Unipartite_df_processed = filter_dataframe_index(Unipartite_df, processed_networks)

# Now we want to join the two uniprtite dataframes together
# on the index (which is the network name)
df_uni_real = join_dataframes(df_uni_real, Unipartite_df_processed)

# get number of unipartite where rchi is less than 2

print(df_uni_real)

# Get every column header that starts with Tag
tag_columns = [col for col in df_uni_real.columns if col.startswith('Tag')]
# Find number of True in each of these tag columns and get as new dataframe
df_uni_real_tags = df_uni_real[tag_columns].sum(axis=0)
# Sort by number of True
df_uni_real_tags = df_uni_real_tags.sort_values(ascending=False)
#print df_uni_real_tags

# remove Tag-Unweighted, and all those with 0
df_uni_real_tags = df_uni_real_tags.drop(['Tag-Unweighted'])
df_uni_real_tags = df_uni_real_tags[df_uni_real_tags > 0]

# find mean rchi for each tag and add it to the tag datafram
df_uni_real_tags = df_uni_real_tags.to_frame()
mean_rchi = []
for tag in df_uni_real_tags.index.values.tolist():
    mean_rchi.append(df_uni_real[df_uni_real[tag] == True]['r^2'].median())
df_uni_real_tags['median_r^2'] = mean_rchi
print(df_uni_real_tags)


# Filter for rchi > 3
df_uni_real_bad = df_uni_real[df_uni_real['r^2']<0.3]
# Sort by rchi
df_uni_real_bad = df_uni_real_bad.sort_values(by=['r^2'], ascending=True)

# Display index, N, E, rchi, density
df_uni_real_bad = df_uni_real_bad[['N', 'E', 'rchi','r^2', 'density','av_counts','hashimoto_radius']]
print(df_uni_real_bad)

'''
corr = df_uni_real_bad.corr(method='pearson')
print(corr)
plot_correlation_matrix(corr)


plt.plot(df_uni_real_bad['hashimoto_radius'], df_uni_real_bad['av_counts'], 'o', color='black')
plt.xlabel('Hashimoto Radius')
plt.ylabel('Average Counts')
plt.show()
'''

# Remove netwoworkd starting with u or U
'''
df_uni_real = df_uni_real[~df_uni_real.index.str.startswith('malaria')]
df_uni_real = df_uni_real[~df_uni_real.index.str.startswith('celegans')]
df_uni_real = df_uni_real[~df_uni_real.index.str.startswith('ugandan')]
df_uni_real = df_uni_real[~df_uni_real.index.str.startswith('us')]
df_uni_real = df_uni_real[~df_uni_real.index.str.startswith('urban')]
'''
uni_names = df_uni_real.index.values.tolist()

# filter for rchi > or equal to 1
# print those that are less than 1
df_uni_fail = df_uni_real[df_uni_real['rchi'] < 1]
print(df_uni_fail)
df_uni_real = df_uni_real[df_uni_real['rchi'] >= 1]

rchis_valid = df_uni_real['rchi'].values.tolist()

# take those for density < 0.1
df_uni_denisty = df_uni_real[df_uni_real['density'] < 0.1]

# take rchis for density < 0.1
rchis_density = df_uni_denisty['rchi'].values.tolist()

# take those for clustering < 0.3
df_uni_clustering = df_uni_real[df_uni_real['clustering'] < 0.3]

# take rchis for clustering < 0.3
rchis_clustering = df_uni_clustering['rchi'].values.tolist()

# find cumulative amount less than rchi

test_rchis = np.linspace(1,max(rchis_valid),100000)

def get_rchi_percentages(rchis, test_rchis=None):
    max_rchi = max(rchis)
    if test_rchis is None:
        test_rchis = np.linspace(1, max_rchi, 100000)
    num_valid = len(rchis)

    counts = [] 
    percentages = []
    for i in test_rchis:
        count = 0
        for j in rchis:
            if j < i:
                count += 1
        percentages.append(count/num_valid)
        counts.append(count)
    return test_rchis, percentages, counts

test_rchis, percentages, counts = get_rchi_percentages(rchis_valid, test_rchis=test_rchis)
test_rchis_density, percentages_density, counts_density = get_rchi_percentages(rchis_density, test_rchis=test_rchis)
test_rchis_clustering, percentages_clustering, counts_clustering = get_rchi_percentages(rchis_clustering,  test_rchis=test_rchis)

# plot cumulative amount less than rchi
plt.plot(test_rchis, percentages,color='black', label='All Networks, Total = '+str(len(rchis_valid)))
plt.plot(test_rchis_density, percentages_density,color='red', label='Density < 0.1, Total = '+str(len(rchis_density)))
plt.plot(test_rchis_clustering, percentages_clustering,color='blue', label='Clustering < 0.3, Total = '+str(len(rchis_clustering)))
rchi = r'$\chi^{2}_{r}$'
plt.xlabel(rchi+' Level')
plt.ylabel('Fraction of Networks with '+rchi +' < ' + rchi + ' Level')
plt.xlim(1, 3)
plt.ylim(0, 1)
#plt.xscale('log')
#fill between different rchi levels
plt.fill_between(test_rchis, percentages, color='gray', alpha=0.2)
plt.fill_between(test_rchis_density, percentages_density,percentages, color='red', alpha=0.2)
plt.fill_between(test_rchis_clustering, percentages_clustering,percentages_density, color='blue', alpha=0.2)
plt.legend(loc='lower right', fontsize=10)
plt.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.13)
plt.savefig('ReportPlots/rchi_percentages.png', dpi=1200)
plt.show()

# plot density vs clustering
# color by r^2
# remove with rchi > 4
df_uni_real_good= df_uni_real[df_uni_real['rchi'] < 4]
plt.figure(figsize=(6,3.5))
plt.scatter(df_uni_real_good['N'], df_uni_real_good['av_degree'], c=df_uni_real_good['rchi'], cmap='viridis')
plt.xlabel('N')
plt.ylabel('av_degree')
plt.colorbar(label=r'$\chi^{2}_{r}$')
plt.xscale('log')
plt.yscale('log')
plt.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.13)
plt.show()


#plot r^2 vs clustering

plt.figure(figsize=(6,3.5))
plt.scatter(df_uni_real['r^2'], df_uni_real['clustering'], c=df_uni_real['density'], cmap='viridis')
plt.xlabel(r'$r^{2}$')
plt.ylabel('Clustering')

plt.colorbar(label='Density')
plt.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.13)
plt.show()



r2s_valid = df_uni_real['r^2'].values.tolist()

# take those for density < 0.1
df_uni_denisty = df_uni_real[df_uni_real['density'] < 0.1]

# take rchis for density < 0.1
r2s_density = df_uni_denisty['r^2'].values.tolist()

# take those for clustering < 0.3
df_uni_clustering = df_uni_real[df_uni_real['clustering'] < 0.1]

# Take union of density and clustering
#df_uni_density_clustering = df_uni_real[(df_uni_real['density'] < 0.05) | (df_uni_real['clustering'] < 0.3)]

# take rchis for clustering < 0.3
r2s_clustering = df_uni_clustering['r^2'].values.tolist()



#r2s_density_clustering = df_uni_density_clustering['r^2'].values.tolist()

# find cumulative amount less than rchi

test_r2s = np.linspace(0,max(r2s_valid),100000)[::-1]

def get_rchi_percentages(r2s, test_r2s=None):
    max_r2 = max(r2s)
    if test_r2s is None:
        test_r2s = np.linspace(0, max_r2, 100000)[::-1]
    num_valid = len(r2s)

    counts = [] 
    percentages = []
    for i in test_r2s:
        count = 0
        for j in r2s:
            if j < i:
                count += 1
        percentages.append(count/num_valid)
        counts.append(count)
    return test_r2s, percentages, counts

test_r2s, percentages, counts = get_rchi_percentages(r2s_valid, test_r2s=test_r2s)
test_r2s_density, percentages_density, counts_density = get_rchi_percentages(r2s_density, test_r2s=test_r2s)
test_r2s_clustering, percentages_clustering, counts_clustering = get_rchi_percentages(r2s_clustering,  test_r2s=test_r2s)
#test_r2s_density_clustering, percentages_density_clustering, counts_density_clustering = get_rchi_percentages(r2s_density_clustering,  test_r2s=test_r2s)

# plot cumulative amount less than rchi
plt.plot(test_r2s, percentages,color='black', label='All Networks, Total = '+str(len(r2s_valid)))
plt.plot(test_r2s_density, percentages_density,color='red', label='Density < 0.1, Total = '+str(len(r2s_density)))
plt.plot(test_r2s_clustering, percentages_clustering,color='blue', label='Clustering < 0.3, Total = '+str(len(r2s_clustering)))
#plt.plot(test_r2s_density_clustering, percentages_density_clustering,color='green', label='Density < 0.1 or Clustering < 0.3, Total = '+str(len(r2s_density_clustering)))
rchi = r'$r^{2}$'
plt.xlabel(rchi+' Level')
plt.ylabel('Fraction of Networks with '+rchi +' < ' + rchi + ' Level')
#plt.xlim(0, 1)
#plt.ylim(0, 1)
#plt.xscale('log')
#fill between different rchi levels
'''
plt.fill_between(test_r2s, percentages, color='gray', alpha=0.2)
plt.fill_between(test_r2s_density, percentages_density,percentages, color='red', alpha=0.2)
plt.fill_between(test_r2s_clustering, percentages_clustering,percentages_density, color='blue', alpha=0.2)
'''
plt.legend(loc='lower right', fontsize=10)
plt.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.13)
plt.savefig('ReportPlots/rchi_asdpercentages.png', dpi=1200)
plt.show()







