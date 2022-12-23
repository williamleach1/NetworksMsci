import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load in analysed dataframes

df_uni_real = pd.read_pickle('Output/RealUniNets/RealUniNets.pkl')

df_bi_real = pd.read_pickle('Output/RealBipartiteNets/RealBipartiteNets.pkl')

# get number of unipartite where rchi is less than 2

uni_names = df_uni_real.index.values.tolist()
rchis = df_uni_real['rchi'].values.tolist()

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












