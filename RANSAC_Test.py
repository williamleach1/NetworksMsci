import numpy as np
import graph_tool.all as gt
from graph_tool import topology
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import ProcessBase as pb


#celegans_interactomes/Microarray
g = pb.load_graph('celegans_interactomes/Microarray')

pos = gt.sfdp_layout(g)
print('done')

k, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k= pb.process(g,1,Real = True, Name = 'celegans_interactomes_Microarray')

ks, inv_c_mean, errs, stds, counts   = pb.unpack_stat_dict(statistics_dict)

plt.figure()
plt.plot(k, 1/c, 'o', color='black', alpha = 0.2)
plt.errorbar(ks, inv_c_mean, yerr=errs, fmt='x', color='red')
plt.xscale('log')
plt.show()

# Now we can do the RANSAC sequentially
v = g.get_vertices()
v = np.array(v)
inv_cs = 1/c
log_k = np.log(k)

v_group = g.new_vertex_property("int")

current_outliers = None
num_outliers = len(log_k)
temp_log_k = log_k
temp_inv_cs = inv_cs
temp_v = v

print(temp_v)

counts = []



plt.figure()
i = 0
#while num_outliers > 0.1*len(log_k):
for i in range(1):
    ransac = RANSACRegressor(residual_threshold = 1.8,max_trials=1000000)
    ransac.fit(temp_log_k.reshape(-1,1), temp_inv_cs)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    count = 0
    for j in range(len(inlier_mask)):
        if inlier_mask[j]:
            v_group[temp_v[j]]=i+1
            count += 1
        else:
            v_group[temp_v[j]]=0
    counts.append(count)
    print('Group ', i , ' has ', count, 'members')

    plt.plot(np.exp((temp_log_k[inlier_mask])), temp_inv_cs[inlier_mask], 'o',color='darkturquoise', alpha = 0.5)

    temp_log_k = temp_log_k[outlier_mask]
    temp_inv_cs = temp_inv_cs[outlier_mask]
    temp_v = temp_v[outlier_mask]
    current_outliers = outlier_mask
    num_outliers = len(temp_log_k)
    #i += 1
 
plt.plot(np.exp(temp_log_k), temp_inv_cs, 'o', color='gold', alpha = 0.5)
plt.xscale('log')
# Add padding between the axes and the axis labels

plt.ylabel(r'$\dfrac{1}{c}$', fontsize=20, rotation=0, labelpad=20)
plt.xlabel(r'$k$', fontsize=20)
plt.legend(['Inliers', 'Outliers'], fontsize=15)
plt.tight_layout()
plt.savefig('ransac_result_graph.png',dpi=900)
plt.show()

g.vertex_properties['inlier'] = v_group

inlier = g.vertex_properties['inlier']

gt.graph_draw(g, pos=pos, vertex_fill_color=inlier,output='ransac_result.svg')

