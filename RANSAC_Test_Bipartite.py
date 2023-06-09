import numpy as np
import graph_tool.all as gt
from graph_tool import topology
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import ProcessBase as pb
from graph_tool import topology
from sklearn.cluster import SpectralClustering
from uncertainties import ufloat, umath

plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (3.5, 2.8),
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.98,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.top': 0.98,
    'axes.labelsize': 12,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.5,
    'lines.markersize': 2,
})

import numpy as np
import seaborn as sns
from graph_tool import GraphView

# Reset the original Matplotlib style
sns.reset_orig()

# Set a specific Seaborn style
sns.set_style('whitegrid')

def calculate_edge_densities(g, partition):
    n = g.num_vertices()
    m = g.num_edges()

    intra_edges_0 = 0
    intra_edges_1 = 0
    inter_edges_01 = 0
    inter_edges_10 = 0
    
    for e in g.edges():
        source, target = int(e.source()), int(e.target())
        if partition[source] == 0 and partition[target] == 0:
            intra_edges_0 += 1
        elif partition[source] == 1 and partition[target] == 1:
            intra_edges_1 += 1
        elif partition[source] == 0 and partition[target] == 1:
            inter_edges_01 += 1
    
    # Calculate the edge densities and other metrics
    density_0_0 = intra_edges_0 #/ max_intra_edges_0
    density_1_1 = intra_edges_1 #/ max_intra_edges_1
    density_0_1 = inter_edges_01 #/ max_inter_edges_01
    density_1_0 = inter_edges_01 #/ max_inter_edges_10

    # Create an extended matrix with the additional values
    matrix = np.array([
        [int(density_0_0), int(density_0_1)],
        [int(density_0_1), int(density_1_1)],
    ])

    return matrix

def sigmoid(x, k=10):
    return 1 / (1 + np.exp(-k * (x - 0.5)))

def branching_factor_similarity_matrix(local_bf_list):
    num_nodes = len(local_bf_list)

    # Create an empty matrix
    similarity_matrix = np.zeros((num_nodes, num_nodes))

    # Calculate the maximum difference between any two nodes' branching factors
    max_diff = max(abs(a - b) for a in local_bf_list for b in local_bf_list)

    # Fill the matrix with the transformed and normalized absolute difference of the branching factors of each pair of nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            abs_diff = abs(local_bf_list[i] - local_bf_list[j])
            normalized_diff = abs_diff / max_diff

            # Calculate the similarity value
            similarity_value = 1 - normalized_diff

            # Apply the sigmoid transformation to emphasize the contrast between similarity values
            transformed_value = sigmoid(similarity_value)

            # Set the similarity value
            similarity_matrix[i, j] = transformed_value

    return similarity_matrix

def rbf(dist_matrix, delta):
    return np.exp(- dist_matrix ** 2 / (2. * delta ** 2))

def dist_matrix(local_bf_list):
    num_nodes = len(local_bf_list)

    # Create an empty matrix
    dist_matrix = np.zeros((num_nodes, num_nodes))

    # fill with absolute differences
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i, j] = abs(local_bf_list[i] - local_bf_list[j])
    
    delta = np.median(dist_matrix)
    transformed_matrix = rbf(dist_matrix, delta)

    return transformed_matrix

def create_density_matrix(g, partition):
    n = g.num_vertices()
    density_matrix = np.zeros((n, n))
    
    for e in g.edges():
        source, target = int(e.source()), int(e.target())
        density_matrix[source, target] = 1
        density_matrix[target, source] = 1  # Assuming the graph is undirected
    
    # Reorder the density matrix based on the partition labels
    sorted_indices = np.argsort(partition)
    density_matrix = density_matrix[sorted_indices, :]
    density_matrix = density_matrix[:, sorted_indices]
    
    return density_matrix

#'celegans_interactomes/Microarray'
if __name__ == '__main__':
    #celegans_interactomes/Microarray
    #Name = 'celegans_interactomes/Microarray'
    #Name = 'celegans_interactomes/Microarray'
    Name = 'marvel_universe'
    g = pb.load_graph(Name)

    test, part, odd = topology.is_bipartite(g,partition=True, find_odd_cycle=True)

    print(test)
    print(part)
    print(odd)
    # access porperty name
    #print(g.vp.name)

# Iterate through the vertices and print their index and name
    
    pos = gt.sfdp_layout(g)
    print('done')

    vertex_similarity = topology.vertex_similarity(g, sim_type="jaccard")

    # Get the number of vertices
    num_vertices = g.num_vertices()


    if '/' in Name:
        name = Name.split('/')
        name = name[0]+'_'+name[1]
    else:
        name = Name

    k, c,popt,pcov, rchi, r, rp, rs, rsp, statistics_dict, mean_k= pb.process(g,1,Real = True, Name = name)

    ks, inv_c_mean, errs, stds, counts   = pb.unpack_stat_dict(statistics_dict)

    unq_dist, mean_count, std_counts, err_counts, all_dist, all_count = pb.process_BFS(g, Real = True, Name = name, give_all=True)  

    # all_dist : list of np.arrays
    #     Distances from the vertex for each vertex
    #     i.e. distances[i] is the distances from vertex i
    # all_count : list of np.arrays
    #     Number of vertices at each distance
    #     i.e. counts[i] is the counts at each distance from vertex i
    
    # find local z for each vertex
    # do this by fitting first 5 points to exponential counts = 1*exp(k*dist)
    # can do this by taking log of counts and fitting to k*dist
    # then take inverse of k to get z
    # if there are less than 5 points, fit to all points
    # do using polyfit

    # if z array exists, load it
    # else, calculate it and save it
    '''
    try:
        jaccard_matrix = np.load('jaccard_matrix'+name+'.npy')
        print('jaccard loaded')
    except:
        vertex_similarity = topology.vertex_similarity(g, sim_type="jaccard")

        # Initialize an empty matrix
        jaccard_matrix = np.zeros((num_vertices, num_vertices))

        # Fill the matrix with values from the vertex property map
        for v in g.vertices():
            jaccard_matrix[int(v)] = vertex_similarity[v]

        # Ensure the matrix is symmetric (as Jaccard similarity should be)
    
        jaccard_matrix = np.maximum(jaccard_matrix, jaccard_matrix.T)

        np.save('jaccard_matrix'+name, jaccard_matrix)
    '''
    try:
        local_z = np.load('local_z'+name+'.npy')
        print('z loaded')
    except:
        local_z = []
        for i in range(len(all_dist)):
            if len(all_dist[i]) < 4:
                weights = []
                for j in range(len(all_dist[i])):
                    weights.append(1/(j+1))
                local_z.append(np.polyfit(all_dist[i], np.log(all_count[i]), 1, w=weights)[0])
            else:
                weights = [1]
                for j in range(1,4):
                    weights.append(1/(j+1))
                local_z.append(np.polyfit(all_dist[i][0:4], np.log(all_count[i][0:4]), 1, w = weights)[0])
        local_z = np.exp(np.array(local_z))
        np.save('local_z'+name, local_z)

    # now calculate the branching factor similarity matrix
    #
    '''
    try:
        bf_matrix = np.load('bf_matrix'+name+'.npy')
    except:
        bf_matrix = dist_matrix(local_z)
        np.save('bf_matrix'+name, bf_matrix)
    
    #combined_matrix = alpha_weight * bf_matrix + (1 - alpha_weight) * jaccard_matrix
    combined_matrix = np.sqrt(bf_matrix * jaccard_matrix)
    print(combined_matrix)
    print(np.min(bf_matrix))
    '''
    # now for spectral clustering
    try:
        spectral_labels = np.load('spectral_labels'+name+'.npy')
        print('spectral loaded')
    except:
        print('spectral not loaded')
        spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed')
        spectral_labels = spectral_clustering.fit_predict(jaccard_matrix)#combined_matrix)
        np.save('spectral_labels'+name, spectral_labels)

    
    partition = spectral_labels  # Replace 'partition' with your partition array

    name = g.vertex_properties["name"]
    names_0 = []
    names_1 = []
    for v in g.vertices():
        #print(f"Vertex {g.vertex_index[v]} has name: {name[v]}")
        if partition[int(v)] == 0:
            names_0.append(name[int(v)])
        else:
            names_1.append(name[int(v)])
    print(names_0[0:100])
    print(names_1[0:100])

    '''
    density_matrix = create_density_matrix(g, partition)

    # Visualize the density matrix using Seaborn's heatmap
    sns.set()
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(density_matrix, cmap='coolwarm', square=True, cbar=False, xticklabels=False, yticklabels=False)

    # Add group labels
    group_0_count = np.sum(partition == 0)
    group_1_count = np.sum(partition == 1)

    plt.text(0.5 * group_0_count, -0.05 * group_0_count, 'Group 0', fontsize=12, ha='center', va='center', transform=ax.transData)
    plt.text(group_0_count + 0.5 * group_1_count, -0.05 * group_1_count, 'Group 1', fontsize=12, ha='center', va='center', transform=ax.transData)

    plt.text(-0.05 * group_0_count, 0.5 * group_0_count, 'Group 0', fontsize=12, ha='center', va='center', rotation='vertical', transform=ax.transData)
    plt.text(-0.05 * group_1_count, group_0_count + 0.5 * group_1_count, 'Group 1', fontsize=12, ha='center', va='center', rotation='vertical', transform=ax.transData)

    plt.title('Density Matrix')
    plt.show()
    '''
    extended_matrix = calculate_edge_densities(g, partition)

    # Extract the edge densities and other metrics
    edge_densities = extended_matrix
    
    # Visualize the edge densities using Seaborn's heatmap
    sns.set()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(edge_densities, cmap='coolwarm', square=True, annot=True, fmt='.4f', cbar=False,
                    xticklabels=['Group 0', 'Group 1'], yticklabels=['Group 0', 'Group 1'])

    plt.title('Edge Densities')
    plt.xlabel('Source Group')
    plt.ylabel('Target Group')
    plt.show()
    
    nodes_group_0 = np.sum (partition == 0)
    nodes_group_1 = np.sum (partition == 1)

    ks_group_0 = k[partition == 0]
    ks_group_1 = k[partition == 1]

    k_mean_group_0 = np.mean(ks_group_0)
    k_mean_group_1 = np.mean(ks_group_1)


    # Print out the additional information
    print(f"Total Nodes for Group 0: {nodes_group_0}")
    print(f"Total Nodes for Group 1: {nodes_group_1}")
    print(f"Average Degree for Group 0: {k_mean_group_0}")
    print(f"Average Degree for Group 1: {k_mean_group_1}")
    print(edge_densities)

    plt.figure(figsize=(6, 6))
    plt.plot(k[spectral_labels == 0], 1/c[spectral_labels == 0], '.', color='blue', alpha=0.2, label='Cluster 1')
    plt.plot(k[spectral_labels == 1], 1/c[spectral_labels == 1], '.', color='red', alpha=0.2, label='Cluster 2')
    plt.xscale('log')
    plt.show()
    
    plt.figure(figsize=(6,3))
    fit_ks = np.linspace(ks[0],ks[-1],100)
    fit_inv_cs = pb.Tim(fit_ks, *popt)

    plt.plot(k, 1/c, '.', color='grey', alpha = 0.2, label='Unaggregated Data')
    plt.errorbar(ks, inv_c_mean, yerr=errs, fmt='o', color='red',ecolor='black' , capsize=3, elinewidth=1, markersize=1, markeredgewidth=1, label='Aggregated Data')

    plt.plot(fit_ks, fit_inv_cs, '--' ,color='blue', label='Fit', linewidth=2)
    plt.xlabel(r'$k$', labelpad=0)
    plt.ylabel(r'$\dfrac{1}{c}$', rotation=0, labelpad=15)
    plt.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.18)
    plt.xscale('log')
    plt.legend()
    plt.savefig('ReportPlots/CelegansUnaggMean.png', dpi=600)
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

    z_groups = []

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


        stat_dict = pb.aggregate_dict(np.exp(temp_log_k[inlier_mask]), temp_inv_cs[inlier_mask])
        temp_ks, temp_inv_c_mean, temp_errs, temp_stds, temp_counts   = pb.unpack_stat_dict(stat_dict)
        #plt.errorbar(temp_ks, temp_inv_c_mean, yerr=temp_errs, fmt='o', color='red',ecolor='black' , capsize=3, elinewidth=1, markersize=1, markeredgewidth=1, label='Aggregated Data')
        fit_coeffs, fit_cov = pb.fitter(np.exp(temp_log_k[inlier_mask]), temp_inv_cs[inlier_mask], pb.Tim)
        print('fit_coeffs = ', fit_coeffs)
        fit_ks = np.linspace(temp_ks[0],temp_ks[-1],100)
        fit_inv_cs = pb.Tim(fit_ks, *fit_coeffs)

        # gradient = -1/ln(z)
        # z = e^(-1/gradient)
        inv_ln_z = fit_coeffs[0]
        inv_ln_z_err = np.sqrt(fit_cov[0][0])
        inv_ln_z_u = ufloat(inv_ln_z, inv_ln_z_err)

        ln_z_u = 1/inv_ln_z_u
        z_u = umath.exp(ln_z_u)
        z_fit_inliers = z_u.n
        z_fit_inliers_err = z_u.s

        print('z_fit_inliers = ', z_fit_inliers)


        plt.plot(np.exp((temp_log_k[inlier_mask])), temp_inv_cs[inlier_mask], 'o',color='darkturquoise', alpha = 0.5, label='Inliers')
        
        plt.plot(fit_ks, fit_inv_cs, '--' ,color='blue', label='Inliers Fit', linewidth=2)
        
        local_z_inliers = local_z[inlier_mask]

        temp_log_k = temp_log_k[outlier_mask]
        temp_inv_cs = temp_inv_cs[outlier_mask]
        temp_v = temp_v[outlier_mask]
        current_outliers = outlier_mask
        num_outliers = len(temp_log_k)

        z_groups.append(local_z_inliers)
        #i += 1
    

    local_z_outliers = local_z[current_outliers]
    
    z_groups.append(local_z_outliers)

    plt.plot(np.exp(temp_log_k), temp_inv_cs, 'o', color='gold', alpha = 0.5, label='Outliers')
    plt.xscale('log')
    # Add padding between the axes and the axis labels

    stat_dict = pb.aggregate_dict(np.exp(temp_log_k), temp_inv_cs)
    temp_ks, temp_inv_c_mean, temp_errs, temp_stds, temp_counts   = pb.unpack_stat_dict(stat_dict)
    #plt.errorbar(temp_ks, temp_inv_c_mean, yerr=temp_errs, fmt='o', color='red',ecolor='black' , capsize=3, elinewidth=1, markersize=1, markeredgewidth=1, label='Aggregated Data')
    fit_coeffs, fit_cov = pb.fitter(np.exp(temp_log_k), temp_inv_cs, pb.Tim)
    inv_ln_z = fit_coeffs[0]
    inv_ln_z_err = np.sqrt(fit_cov[0][0])
    inv_ln_z_u = ufloat(inv_ln_z, inv_ln_z_err)

    ln_z_u = 1/inv_ln_z_u
    z_u = umath.exp(ln_z_u)
    z_fit_outliers = z_u.n
    z_fit_outliers_err = z_u.s

    print('z_fit_outliers = ', z_fit_outliers)

    fit_ks = np.linspace(temp_ks[0],temp_ks[-1],100)
    fit_inv_cs = pb.Tim(fit_ks, *fit_coeffs)
    plt.plot(fit_ks, fit_inv_cs, '--' ,color='red', label='Outliers Fit', linewidth=2)


    plt.ylabel(r'$\dfrac{1}{c}$', rotation=0, labelpad=20)
    plt.xlabel(r'$k$', rotation=0, labelpad=0)
    plt.legend()
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.13)
    plt.savefig('ReportPlots/ransac_result_graph.png',dpi=900)
    plt.show()

    g.vertex_properties['inlier'] = v_group

    inlier = g.vertex_properties['inlier']

    gt.graph_draw(g, pos=pos, vertex_fill_color=inlier,output='ReportPlots/ransac_result.png')

    # New figure for histograms of local z
    plt.figure()
    plt.hist(z_groups[0], bins=25, label='Inliers', histtype='step', density=True)
    plt.hist(z_groups[1], bins=25, label='Outliers', histtype='step', density=True)
    # plot vertical lines at the fitted z values
    #plt.axvline(z_fit_inliers, color='blue', linestyle='--', label='Inliers Fit')
    #plt.axvline(z_fit_outliers, color='red', linestyle='--', label='Outliers Fit')

    plt.xlabel('Local z')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

