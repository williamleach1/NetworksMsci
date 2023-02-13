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

corr = df_uni_real_bad.corr(method='pearson')
print(corr)
plot_correlation_matrix(corr)


plt.plot(df_uni_real_bad['hashimoto_radius'], df_uni_real_bad['av_counts'], 'o', color='black')
plt.xlabel('Hashimoto Radius')
plt.ylabel('Average Counts')
plt.show()


uni_names = df_uni_real.index.values.tolist()
rchis = df_uni_real['rchi'].values.tolist()
densities = df_uni_real['density'].values.tolist()
# print those less than 1 - remove from list
valid_uni = []
valid_rchi = []
for i in range(len(uni_names)):
    if rchis[i] < 1:
        print(uni_names[i], ' removed from list as rchi is 0 indicating faiure :', rchis[i])
    else:
        valid_uni.append(uni_names[i])
        valid_rchi.append(rchis[i])

max_rchi = max(valid_rchi)
# find cumulative amount less than rchi

test_rchis = np.linspace(1, max_rchi, 100000)
num_valid = len(valid_rchi)

counts = [] 
percentages = []
for i in test_rchis:
    count = 0
    for j in valid_rchi:
        if j < i:
            count += 1
    percentages.append(count/num_valid)
    counts.append(count)

# plot cumulative amount less than rchi
plt.plot(test_rchis, percentages,color='black')
rchi = r'$\chi^{2}_{r}$'
plt.xlabel(rchi+' Level')
plt.ylabel('Fraction of networks with '+rchi +' < ' + rchi + ' Level')
plt.xlim(1, 10)
plt.ylim(0, 1)
#plt.xscale('log')
plt.fill_between(test_rchis, percentages, step="pre", alpha=0.4)
plt.show()












