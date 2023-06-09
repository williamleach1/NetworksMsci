'''
Placeholders for the analysis of real bipartite graph results
'''

import pandas as pd
from AnalysisBase import *
from uncertainties import unumpy as unp
from numpy import exp

df_bipartite = pd.read_pickle('Output/RealBipartiteNets/RealBipartiteNets_uncon.pkl')

print(df_bipartite[['rchi_uni', 'rchi_1', 'rchi_2']])



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
        ax.scatter([i], [row['rchi_uni']], c='grey')  # Low end of color scale
        ax.scatter([i], [row['rchi_2']], c='blue')  # Low end of color scale
    else:
        ax.scatter([i], [row['rchi_1']], c='blue')  # Low end of color scale
        ax.scatter([i], [row['rchi_uni']], c='grey')  # High end of color scale
        ax.scatter([i], [row['rchi_2']], c='red')  # High end of color scale

    ys = [row['rchi_1'], row['rchi_2'], row['rchi_uni']]
    # sort ys
    ys.sort()

    ax.plot([i, i], [row['rchi_1'], row['rchi_2']], 'k--') 

labels = df.index

# if it contains '/' then split and take the second part
labels = [label.split('/')[1] if '/' in label else label for label in labels]


# Set x-ticks to be the DataFrame index, rotate them by 90 degrees, and set the font size to 8
plt.xticks(ticks=range(len(df)), labels=labels, rotation=45, fontsize=9, ha='right')

plt.ylim(0,1)

ylabel = r'$r^{2}$'#r'$\dfrac{(\chi^{2}_{r,ab} - \chi^{2}_{r,uni})}{ \chi^{2}_{r,uni}}$'

# legend at the top custom grey - Unipartite, Red - larger N, Blue - smaller N
import matplotlib.patches as mpatches
grey_patch = mpatches.Patch(color='grey', label='Unipartite')
red_patch = mpatches.Patch(color='red', label='Larger Group')
blue_patch = mpatches.Patch(color='blue', label='Smaller Group')
ax.legend(handles=[grey_patch, red_patch, blue_patch], loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, fancybox=True, fontsize=9)


#plt.xlabel('Network Name')
plt.ylabel(ylabel, fontsize=11)
plt.subplots_adjust(right=0.98,top=0.92,bottom=0.32, left=0.12)
plt.savefig('ReportPlots/BipartiteReal/rchivsrchiuni_uncon.png', dpi=900)
plt.show()


# Take 'a' and 'b' from df

gradient_df = df[['grad_1', 'grad_2', 'grad_1_error', 'grad_2_error', 'N_1','N_2']]

df = gradient_df

# add new columns for the larger and smaller group gradients
# Determine the condition
condition = df['N_1'] > df['N_2']
# Add new columns for the larger and smaller group gradients
df['larger_group_grad'] = np.where(condition, df['grad_1'], df['grad_2'])
df['larger_group_grad_error'] = np.where(condition, df['grad_1_error'], df['grad_2_error'])
df['smaller_group_grad'] = np.where(condition, df['grad_2'], df['grad_1'])
df['smaller_group_grad_error'] = np.where(condition, df['grad_2_error'], df['grad_1_error'])
# Plot the larger group gradient against the smaller group gradient
plt.errorbar(df['larger_group_grad'], df['smaller_group_grad'], xerr=df['larger_group_grad_error'], yerr=df['smaller_group_grad_error'], fmt='o', capsize=2, elinewidth=0.5,mfc='none', markersize=3, color='black', ecolor='blue', alpha=1)
plt.xlabel(r'$ \dfrac{2}{\ln(\bar{z}_{a}\bar{z}_{b})}$' +'  (for Larger Group)')
plt.ylabel(r'$ \dfrac{2}{\ln(\bar{z}_{a}\bar{z}_{b})}$' +'  (for Smaller Group)')
# plot a stright line y=x
ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
plt.plot([0,max(ymax,xmax)], [0,max(ymax,xmax)], 'k--')
plt.subplots_adjust(right=0.98,top=0.98,bottom=0.12, left=0.12)
plt.savefig('ReportPlots/BipartiteReal/grad1vsgrad2_uncon.png', dpi=900)
plt.show()


print(df)

plt.figure(figsize=(8,5))



# Define some colors and markers
colors = plt.cm.rainbow(np.linspace(0, 1, 6))  # 6 different colors
markers = ['o', 'v', '^', 's']  # 4 different markers
labels = df.index

# if it contains '/' then split and take the second part
labels = [label.split('/')[1] if '/' in label else label for label in labels]

# set labels as the index
df.index = labels


# Plot each point separately
for counter, (i, row) in enumerate(df.iterrows()):
    # Calculate the index of the color and marker based on the current combination
    color_index = (counter // len(markers)) % len(colors)
    marker_index = counter % len(markers)

    plt.errorbar(row['larger_group_grad'], row['smaller_group_grad'],
                 xerr=row['larger_group_grad_error'], yerr=row['smaller_group_grad_error'],
                 marker=markers[marker_index], capsize=2, elinewidth=0.5, mfc='none', markersize=7, 
                 color=colors[color_index], ecolor='grey', alpha=1, label=str(i),mew=1.5)

plt.xlabel(r'$ \dfrac{2}{\ln(\bar{z}_{a}\bar{z}_{b})}$' +'  (for Larger Group)')
plt.ylabel(r'$ \dfrac{2}{\ln(\bar{z}_{a}\bar{z}_{b})}$' +'  (for Smaller Group)')

# plot a straight line y=x
ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
plt.plot([0, max(ymax, xmax)], [0, max(ymax, xmax)], 'k--')


plt.ylim(0,1.5)

# Create a legend outside the plot to the right
plt.legend(title='Network Name', bbox_to_anchor=(1.0, 0.5), loc='center left', fontsize=9, fancybox=True)

plt.subplots_adjust(right=0.7,left=0.12, top=0.97,bottom=0.15)  # adjust the right boundary of the plot window
plt.savefig('ReportPlots/BipartiteReal/grad1vsgrad2_uncon.png', dpi=900)
plt.show()






