import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AnalysisBase import *
import seaborn as sns
# Load in analysed dataframes

df_uni_real = pd.read_pickle('Output/RealUniNets/RealUniNets_HO.pkl')

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

# Filter for rchi > 3
df_uni_real_bad = df_uni_real[df_uni_real['rchi'] > 3]
#df_uni_real_bad = df_uni_real_bad[df_uni_real_bad['rchi'] > 1]
# Sort by rchi
df_uni_real_bad = df_uni_real_bad.sort_values(by=['rchi'], ascending=False)

# Display index, N, E, rchi, density
df_uni_real_bad = df_uni_real_bad[['N', 'E', 'rchi', 'density','av_counts','hashimoto_radius']]
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
plt.plot(test_rchis, percentages,color='black', label='All Networks, N = '+str(len(rchis_valid)))
plt.plot(test_rchis_density, percentages_density,color='red', label='Density < 0.1, N = '+str(len(rchis_density)))
plt.plot(test_rchis_clustering, percentages_clustering,color='blue', label='Clustering < 0.3, N = '+str(len(rchis_clustering)))
rchi = r'$\chi^{2}_{r}$'
plt.xlabel(rchi+' Level', fontsize=14)
plt.ylabel('Fraction of networks with '+rchi +' < ' + rchi + ' Level', fontsize=14)
plt.xlim(1, 4)
plt.ylim(0, 1)
#plt.xscale('log')
#fill between different rchi levels
plt.fill_between(test_rchis, percentages, color='gray', alpha=0.2)
plt.fill_between(test_rchis_density, percentages_density,percentages, color='red', alpha=0.2)
plt.fill_between(test_rchis_clustering, percentages_clustering,percentages_density, color='blue', alpha=0.2)
plt.legend(fontsize=16)
plt.savefig('Output/MiscPlots/rchi_percentages.png', dpi=1200)
plt.show()












