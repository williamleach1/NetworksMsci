'''
Placeholders for the analysis of real bipartite graph results
'''

import pandas as pd
from AnalysisBase import *
from uncertainties import unumpy as unp
from numpy import exp

df_bipartite = pd.read_pickle('Output/RealBipartiteNets/RealBipartiteNets.pkl')


print(df_bipartite[['rchi_uni', 'rchi_1', 'rchi_2','rchi']])

# add new_column for min[mean_k_1, mean_k_2]/max[mean_k_1, mean_k_2]
df_bipartite['k_ratio'] = df_bipartite[['mean_k_1', 'mean_k_2']].min(axis=1)/df_bipartite[['mean_k_1', 'mean_k_2']].max(axis=1)

df = df_bipartite

# Create a new column 'a_with_error' combining 'a' and 'an_error'
df['a_with_error'] = unp.uarray(df['a'], df['a_error'])

# Calculate 'z_a' and 'z_a_error' from 'a_with_error'
df['z_a'] = unp.exp(df['a_with_error']) # Inverse of natural log is exponential

# Separate the nominal values and the standard deviations
df['z_a_error'] = unp.std_devs(df['z_a'])
df['z_a'] = unp.nominal_values(df['z_a'])

# Remove 'a_with_error' column if not needed
df.drop('a_with_error', axis=1, inplace=True)

df['b_with_error'] = unp.uarray(df['b'], df['b_error'])

# Calculate 'z_b' and 'z_b_error' from 'b_with_error'
df['z_b'] = unp.exp(df['b_with_error']) # Inverse of natural log is exponential

# Separate the nominal values and the standard deviations
df['z_b_error'] = unp.std_devs(df['z_b'])
df['z_b'] = unp.nominal_values(df['z_b'])

# Remove 'b_with_error' column if not needed
df.drop('b_with_error', axis=1, inplace=True)

df['z_a_with_error'] = unp.uarray(df['z_a'], df['z_a_error'])
df['z_b_with_error'] = unp.uarray(df['z_b'], df['z_b_error'])

# Calculate 'z_ab' and 'z_ab_error' from 'z_a_with_error' and 'z_b_with_error'
df['z_ab'] = df['z_a_with_error'] * df['z_b_with_error']

# Separate the nominal values and the standard deviations
df['z_ab_error'] = unp.std_devs(df['z_ab'])
df['z_ab'] = unp.nominal_values(df['z_ab'])

# Remove 'z_a_with_error' and 'z_b_with_error' columns if not needed
df.drop(['z_a_with_error', 'z_b_with_error'], axis=1, inplace=True)




from uncertainties import ufloat, unumpy
from numpy import log

def calculate_beta(row):
    # Get z_ab, z_ab_error, and N from the row
    z_ab = row['z_ab']
    z_ab_error = row['z_ab_error']
    N = row['N_1'] + row['N_2']

    # Create an uncertain number for z_ab
    z_ab_with_error = ufloat(z_ab, z_ab_error)

    # Calculate beta using the given formula
    beta = (1/(z_ab_with_error-1) + unp.log(z_ab_with_error-1)/unp.log(z_ab_with_error)) + (1/unp.log(z_ab_with_error))*unp.log(N)

    # Return the nominal value and the standard deviation (i.e., the error)
    return pd.Series([beta.nominal_value, beta.std_dev])

# Add 'N' column to the dataframe
df['N'] = df['N_1'] + df['N_2']

# Apply the function to each row and create new columns for 'beta' and 'beta_error'
df[['beta', 'beta_error']] = df.apply(calculate_beta, axis=1)

print(df[['a','b','z_a','z_b','z_a_error','z_b_error','alpha','alpha_error','beta','beta_error']])

# fit alpha and beta to straught line
from scipy.optimize import curve_fit
import numpy as np

def func(x, a, b):
    return a * x + b


xdata = df['alpha']


ydata = df['beta']
y_error = df['beta_error']

popt, pcov = curve_fit(func, xdata, ydata)

print(popt)
print(pcov)
# plot alpha vs beta
fig, ax = plt.subplots(figsize=(5, 4))

plt.plot([0,8],[0,8],'--', color='grey', label=r'$\beta_{fit}=\beta(N, z_{a,fit}z_{b,fit})$')

# plot alpha vs beta with error
plt.errorbar(df['alpha'], df['beta'], xerr=df['alpha_error'], yerr=df['beta_error'], fmt='o', mew=1, capsize=2, elinewidth=1, markersize=3, mfc='none',ecolor = 'red', label='Data')
plt.xlabel(r'$\beta_{fit}$', fontsize=11)
plt.ylabel(r'$\beta(N, z_{a,fit}z_{b,fit})$', fontsize=11)

xdata_ordered = np.sort(xdata)

plt.plot(xdata_ordered, func(xdata_ordered, *popt), 'k--', label='Fit to data: m=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.subplots_adjust(right=0.98,top=0.98,bottom=0.12, left=0.12)
plt.savefig('ReportPlots/BipartiteReal/alpha_vs_beta.png', dpi=900)
plt.show()


import matplotlib.colors as mcolors

# plot rchi vs rchi_uni
# colour by k_ratio
#plt.scatter(df_bipartite['rchi_uni'], df_bipartite['rchi'], c=df_bipartite['k_ratio'], cmap='viridis')
#plt.colorbar()

# plit rchi_uni as x axis, and rchi_1 and rchi_2 as y axis. Connect rchi_1 and rchi_2 with a line with colour k_ratio, and circles at head and tail
# Sort the DataFrame by 'rchi_uni'
df = df_bipartite
# keep only rows with all rchi > 0
df = df[(df[['rchi_1','rchi_2','rchi_uni']] > 0).all(axis=1)]

# Scale rchi_1 and rchi_2 by their difference from rchi_uni
df['rchi_1'] = (df['rchi_1'] - df['rchi_uni']) / df['rchi_uni']
df['rchi_2'] = (df['rchi_2'] - df['rchi_uni']) / df['rchi_uni']

# Create a colormap
cmap = plt.cm.viridis

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(5, 4))

# Plot lines and points
for i, row in df.iterrows():
     # Use a neutral color for the lines
    
    # Color according to whether N_1 is greater than N_2 or not
    if row['N_1'] > row['N_2']:
        ax.scatter([i], [row['rchi_1']], c='red')  # High end of color scale
        ax.scatter([i], [row['rchi_2']], c='blue')  # Low end of color scale
    else:
        ax.scatter([i], [row['rchi_1']], c='blue')  # Low end of color scale
        ax.scatter([i], [row['rchi_2']], c='red')  # High end of color scale

    ax.plot([i, i], [row['rchi_1'], row['rchi_2']], 'k--') 

labels = df.index

# if it contains '/' then split and take the second part
labels = [label.split('/')[1] if '/' in label else label for label in labels]


# Set x-ticks to be the DataFrame index, rotate them by 90 degrees, and set the font size to 8
plt.xticks(ticks=range(len(df)), labels=labels, rotation=45, fontsize=9, ha='right')

plt.ylim(-1,2.5)

ylabel = r'$\dfrac{(\chi^{2}_{r,ab} - \chi^{2}_{r,uni})}{ \chi^{2}_{r,uni}}$'

import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Larger Group')
blue_patch = mpatches.Patch(color='blue', label='Smaller Group')
ax.legend(handles=[red_patch, blue_patch], loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, fancybox=True, fontsize=9)


#plt.xlabel('Network Name')
plt.ylabel(ylabel, fontsize=11)
plt.subplots_adjust(right=0.98,top=0.92,bottom=0.32, left=0.2)
plt.savefig('ReportPlots/BipartiteReal/rchivsrchiuni.png', dpi=900)
plt.show()


# Take 'a' and 'b' from df





