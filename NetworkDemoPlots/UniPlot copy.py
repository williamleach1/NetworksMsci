import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
import matplotlib.cm as cm
import os
import pickle
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
import scipy as sp
from uncertainties import ufloat, umath
import matplotlib.animation as animation
warnings.filterwarnings("ignore")

plt.rcParams.update({
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
})

def E_C(k, a, b):
    

    return -a*np.log(k) + b

def aggregate_over_x(x, y):
    """Aggregate over x to find mean and standard deviation in y"""

    x = np.array(x)
    y = np.array(y)

    x_unique, counts = np.unique(x, return_counts=True)
    indices = np.where(counts >= 2)
    
    x_filtered = x_unique[indices]
    y_mean = np.array([np.mean(y[x == k]) for k in x_filtered])
    y_std = np.array([np.std(y[x == k]) for k in x_filtered])
    counts = counts[indices]
    errors = y_std / np.sqrt(counts)

    return x_filtered, y_mean, errors, y_std, counts


def create_graph():
    G = nx.watts_strogatz_graph(100,4,0.5,seed=42)
    return G

def bfs_tree(G, root):
    return nx.bfs_tree(G, root)

def radial_tree_layout(graph, root, radius=3):
    pos = {}
    visited = set()

    def dfs(node, depth, angle_range):
        visited.add(node)
        n_children = len([n for n in graph.neighbors(node) if n not in visited])
        if depth >= 2:
            angle_step = (angle_range[1] - angle_range[0]) / max(n_children, 1) / (depth - 1)
        else:
            angle_step = (angle_range[1] - angle_range[0]) / max(n_children, 1)

        angle = angle_range[0]
        for child in graph.neighbors(node):
            if child not in visited:
                if depth >= 2:
                    pos[child] = ((depth+1) * radius * np.cos(angle), (depth+1) * radius * np.sin(angle))
                else:
                    pos[child] = ((depth+1) * radius * np.cos(angle), (depth+1) * radius * np.sin(angle))
                dfs(child, depth + 1, (angle - angle_step / 2, angle + angle_step / 2))
                angle += angle_step

    pos[root] = (0, 0)
    dfs(root, 0, (0, 2 * np.pi))

    return pos


def color_nodes_based_on_ring(tree, root, max_distance):
    min_distance = 0
    node_colors = {}
    max_ring = max(nx.shortest_path_length(tree, source=root, target=node) for node in tree.nodes)

    # Choose a colormap, e.g., 'viridis'
    colormap = cm.get_cmap('viridis')
    
    for node in tree.nodes:
        ring = nx.shortest_path_length(tree, source=root, target=node)
        node_colors[node] = colormap((ring - min_distance) / (max_distance - min_distance))

    return node_colors


def interpolate_positions(pos1, pos2, alpha):
    pos = {}
    for node in pos1:
        pos[node] = (pos1[node][0] * (1 - alpha) + pos2[node][0] * alpha,
                     pos1[node][1] * (1 - alpha) + pos2[node][1] * alpha)
    return pos

def interpolate_colors(color1, color2, alpha):
    return tuple(color1[i] * (1 - alpha) + color2[i] * alpha for i in range(3)) + (color1[3],)

def draw_colorbar(cax, cmap, max_distance):
    norm = plt.Normalize(0, max_distance)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax,
                        orientation='vertical',
                        fraction=0.035,
                        pad=0.1)
    cbar.set_ticks(np.arange(0, max_distance + 1))
    cbar.set_ticklabels(np.arange(0, max_distance + 1))
    cbar.set_label('')
    return cbar

def main():
    start = time.time()
    G = create_graph()
    degree = [val for (node, val) in G.degree()]
    closeness = nx.closeness_centrality(G)

    min_degree = min(degree)
    max_degree = max(degree)
    min_closeness = min(closeness.values())
    max_closeness = max(closeness.values())

    roots = [0,1,2,3,4,5,6,7,8,9]
    #roots = [node for node in G.nodes]
    frames_per_transition = 50
    pause_frames = 10

    pos_dir = "positions"
    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)

    trees = [bfs_tree(G, root) for root in roots]
    pos_list = []

    for root, tree in zip(roots, trees):
        pos_file = os.path.join(pos_dir, f"pos_{root}.pickle")
        if os.path.exists(pos_file):
            with open(pos_file, "rb") as f:
                pos = pickle.load(f)
        else:
            pos = radial_tree_layout(tree, root)
            #print('Done root', root)
            with open(pos_file, "wb") as f:
                pickle.dump(pos, f)
        pos_list.append(pos)


    # Calculate the maximum distance from any root
    max_distance = 0
    for tree, root in zip(trees, roots):
        max_distance = max(max_distance, max(nx.shortest_path_length(tree, source=root, target=node) for node in tree.nodes))

    cmap = cm.get_cmap('viridis')
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(nrows=2, ncols=4, width_ratios=[15, 1, 2, 15], height_ratios=[1, 0.2])

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    pax = fig.add_subplot(gs[0, 3])
    pax1 = fig.add_subplot(gs[1, 0])
    pax2 = fig.add_subplot(gs[1, 3])
    plt.subplots_adjust(bottom=0.2)

    ax.set_xlim( - 22, 22)
    ax.set_ylim( - 22, 22)

    cbar = draw_colorbar(cax, cmap, max_distance)

    label = "Distance from Root"
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.labelpad = 15
    cbar.ax.set_ylabel(label, fontsize=20)
    pax2.clear()
    pax1.clear()
    pax1.axis('off')
    pax2.axis('off')

    def animate(frame):

        if frame % 25 == 0:
            os.system('cls' if os.name == 'nt' else 'clear')
    
            print(f"Processing frame {frame + 1}/{(len(roots) - 1) * (frames_per_transition + pause_frames) + pause_frames} ({100 * (frame + 1) / ((len(roots) - 1) * (frames_per_transition + pause_frames) + pause_frames):.2f}%)")
    
        ax.clear()
        pax.clear()
        transition, t = divmod(frame, frames_per_transition + pause_frames)

        if transition >= len(roots) - 1:
            if t < pause_frames:
                root = roots[-1]
                tree = trees[-1]
                pos = pos_list[-1]
                tree_edge_alpha = 1
                non_tree_edge_alpha = 0.4
                tree_edge_width = 3
                non_tree_edge_width = 1
                node_colors1 = color_nodes_based_on_ring(tree, root, max_distance)
            else:
                return
        elif t < pause_frames:
            root = roots[transition]
            tree = trees[transition]
            pos = pos_list[transition]
            tree_edge_alpha = 1
            non_tree_edge_alpha = 0.3
            tree_edge_width = 3
            non_tree_edge_width = 1
            node_colors1 = color_nodes_based_on_ring(tree, root, max_distance)
        else:
            root = roots[transition]
            tree = trees[transition]
            pos = interpolate_positions(pos_list[transition], pos_list[transition + 1], (t - pause_frames) / frames_per_transition)
            node_colors1 = color_nodes_based_on_ring(trees[transition], roots[transition], max_distance)
            node_colors2 = color_nodes_based_on_ring(trees[transition + 1], roots[transition + 1], max_distance)
            for node in tree.nodes:
                node_colors1[node] = interpolate_colors(node_colors1[node], node_colors2[node], (t - pause_frames) / frames_per_transition)

            halfway = frames_per_transition / 2
            if t - pause_frames < halfway:
                tree_edge_alpha = 1 - 0.6 * (t - pause_frames) / halfway
                tree_edge_width = 3 - 2 * (t - pause_frames) / halfway
            else:
                tree = trees[transition + 1]  # Use the new tree after halfway
                tree_edge_alpha = 0.4 + 0.6 * (t - pause_frames - halfway) / halfway
                tree_edge_width = 1 + 2 * (t - pause_frames - halfway) / halfway

            non_tree_edge_alpha = 0.3
            non_tree_edge_width = 1

        G_edges_no_selfloop = [(u, v) for u, v in G.edges if u != v]
        tree_edges_no_selfloop = [(u, v) for u, v in tree.edges if u != v]

        # draw network nodes
        nx.draw_networkx_edges(G, pos, edgelist=G_edges_no_selfloop, alpha=non_tree_edge_alpha, edge_color='black', width=non_tree_edge_width, arrowstyle='-', ax=ax)
        nx.draw_networkx_edges(tree, pos, edgelist=tree_edges_no_selfloop, alpha=tree_edge_alpha, edge_color='black', width=tree_edge_width, arrowstyle='-', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=node_colors1.keys(), node_color=list(node_colors1.values()), node_size=200,linewidths=1, ax=ax)

        ax.axis("off")
        center_circle = patches.Circle((0, 0), radius=0.5, color='red')

        # add cicle patch and bring it to the front
        ax.add_patch(center_circle)

        # Draw circles for each layer
        layers = [0, 1, 2, 3, 4, 5, 6,7]
        for i in range(len(layers)):
            c = plt.Circle((0, 0), i*3, linestyle = '--',color='k', fill=False, alpha = 0.5, clip_on=False)
            ax.add_artist(c)

        ax.set_xlim(-24, 24)
        ax.set_ylim(-24, 24)
        ax.set_aspect('equal', adjustable='box')

        displayed_roots = roots[:transition+1]
        degrees = [degree[root] for root in displayed_roots]
        inv_closenesses = [1/closeness[root] for root in displayed_roots]
        # Find mean inverse closeness for each degree

        
        ks, inv_c_mean, errs, stds, counts   = aggregate_over_x(degrees, inv_closenesses)
        
        pax.errorbar(ks, inv_c_mean, yerr=errs, fmt='x', color='black', capsize=4, capthick=2, elinewidth=2, markersize=7, markeredgewidth=2, alpha = 0.8, label = 'Mean at Degree')
        pax2.axis('off')
        pax1.text(0.25, 0.25, r'$\dfrac{1}{C_r}=-\dfrac{1}{ln(\bar{z})}ln(k_r) +\beta (\bar{z},N)$', transform=pax1.transAxes, fontsize=35, verticalalignment='bottom')
        pax1.axis('off')
        if len(ks) >= 2:
            # start fitting every iteration
            popt, pcov = sp.optimize.curve_fit(E_C , degrees, inv_closenesses, p0=[1, 1])
            all_ks = np.linspace(min_degree, max_degree, 100)
            pax.plot(all_ks, E_C(all_ks, *popt), 'k--', alpha = 0.8, label = 'Fit')
            pax2.clear()
            pax2.axis('off')
            # calculate z, beta and their errors
            inv_ln_z = popt[0]
            beta = popt[1]
            inv_ln_z_err = np.sqrt(pcov[0][0])
            beta_err = np.sqrt(pcov[1][1])
            inv_ln_z_u = ufloat(inv_ln_z, inv_ln_z_err)
            beta_u = ufloat(beta, beta_err)
            beta = beta_u.n
            beta_err = beta_u.s
            ln_z_u = 1/inv_ln_z_u
            z_u = umath.exp(ln_z_u)
            z = z_u.n
            z_err = z_u.s
            # display results of fit
            text = r'$\beta = $' + f'{beta:.2f}' + r'$ \pm $' + f'{beta_err:.2f}' + r'$ \quad \quad \bar{z} = $' + f'{z:.2f}' + r'$ \pm $' + f'{z_err:.2f}'
            pax2.text(0.05, 0.2, text , transform=pax2.transAxes, fontsize=25, verticalalignment='bottom')

        # plot the points
        pax.scatter(degrees[:-1], inv_closenesses[:-1], color='blue', alpha = 0.8, label = 'Prior Nodes')
        pax.scatter(degrees[-1], inv_closenesses[-1], color='red', alpha = 0.8, label = 'Current Node')
        pax.set_xlabel('Degree of Root, '+ r'$k_r$', fontsize=25, labelpad=20)
        pax.set_ylabel('Inverse Closeness ' +  r'$1/C$', fontsize=25, labelpad=20)
        pax.set_xlim(min_degree-0.1, max_degree+0.1)
        pax.set_ylim(1/max_closeness - 0.1, 1/min_closeness+0.1)
        # set ticks to be fontsize 20

        for tick in pax.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        pax.set_xscale('log')
        pax.legend(loc='upper right', fontsize=25)
        for tick in pax.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        pax.grid(True)

    # save animation
    ani = animation.FuncAnimation(fig, animate, frames=(len(roots) - 1) * (frames_per_transition + pause_frames) + pause_frames, interval=15, blit=False, repeat=False)
    plt.tight_layout()
    writer_video = FFMpegWriter(fps=25, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('animation.mp4', writer=writer_video)
    end = time.time()
    print("Time taken to process: ", end - start)

if __name__ == "__main__":
    main()
