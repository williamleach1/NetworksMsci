import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from concurrent.futures import ProcessPoolExecutor
import time
import warnings
from collections import defaultdict
import matplotlib.cm as cm
import os
import pickle
import matplotlib.patches as patches
from collections import deque
import graph_tool.all as gt
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
import ProcessBase as pb
import scipy as sp
from uncertainties import ufloat, umath
warnings.filterwarnings("ignore")


def create_graph():
    #G = nx.Graph()
    #edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (3, 8), (4, 9), (5, 10), (6, 11), (7, 12)]
    #G.add_edges_from(edges)
    G = nx.watts_strogatz_graph(100,4,0.5,seed=42)
    return G

def bfs_tree(G, root):
    return nx.bfs_tree(G, root)


def circular_layout(root, tree, radius=1.0, max_iter=1000, temp=100, cooling_rate=0.995, perturb_angle=0.05, intra_ring_penalty=2.0, cross_penalty=1.0):
    pos = nx.circular_layout(tree, scale=radius, center=(0, 0))

    # Place root at (0, 0)
    pos[root] = (0, 0)
    visited = {node: False for node in tree.nodes}
    level = 1
    current_level_nodes = [root]

    def sorted_next_level_nodes(parent_nodes):
        next_level_nodes = []
        for parent in parent_nodes:
            children = list(tree.successors(parent))
            children = [child for child in children if not visited[child]]
            sorted_children = sorted(children, key=lambda child: pos[parent][1], reverse=True)
            next_level_nodes.extend(sorted_children)
        return next_level_nodes

    def total_distance2(pos, nodes, tree):
        distance = 0
        for node in nodes:
            parent = list(tree.predecessors(node))[0] if node != root else None
            if parent is not None:
                dx, dy = pos[node][0] - pos[parent][0], pos[node][1] - pos[parent][1]
                distance += np.sqrt(dx * dx + dy * dy)

            # Penalty for nodes too close within the same ring
            for other_node in nodes:
                if node != other_node:
                    dx, dy = pos[node][0] - pos[other_node][0], pos[node][1] - pos[other_node][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radius * 0.5:  # You can adjust this threshold based on the desired minimum distance
                        distance += intra_ring_penalty * (radius * 0.5 - dist)

        return distance
    
    def total_distance(pos, nodes, tree):
        distance = 0
        for node in nodes:
            parent = list(tree.predecessors(node))[0] if node != root else None
            if parent is not None:
                dx, dy = pos[node][0] - pos[parent][0], pos[node][1] - pos[parent][1]
                distance += np.sqrt(dx * dx + dy * dy)

            # Penalty for nodes too close within the same ring
            for other_node in nodes:
                if node != other_node:
                    dx, dy = pos[node][0] - pos[other_node][0], pos[node][1] - pos[other_node][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radius * 0.5:  # You can adjust this threshold based on the desired minimum distance
                        distance += intra_ring_penalty * (radius * 0.5 - dist)

            # Penalty for crossing edges
            neighbors = list(tree.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in tree.neighbors(neighbors[i]):
                        x1, y1 = pos[node]
                        x2, y2 = pos[neighbors[i]]
                        x3, y3 = pos[neighbors[j]]
                        if (x2 - x1) * (y3 - y1) > (y2 - y1) * (x3 - x1):
                            distance += cross_penalty

        return distance
    def swap_positions(pos, nodes):
        new_pos = pos.copy()
        node = random.choice(nodes)
        level_radius = np.sqrt(new_pos[node][0]**2 + new_pos[node][1]**2)
        theta = math.atan2(new_pos[node][1], new_pos[node][0])
        delta_theta = random.uniform(-perturb_angle, perturb_angle)
        theta += delta_theta
        new_pos[node] = (level_radius * np.cos(theta), level_radius * np.sin(theta))
        return new_pos

    while current_level_nodes:
        next_level_nodes = sorted_next_level_nodes(current_level_nodes)
        n = len(next_level_nodes)
        for i, node in enumerate(next_level_nodes):
            visited[node] = True
            theta = 2 * np.pi * i / n + np.pi / 2
            pos[node] = ((radius) * (level) * np.cos(theta), (radius) * (level) * np.sin(theta))

        # Adjust the position of the first-degree neighbors if they are at the same position as the root

        # ... (same as before)
        if n > 1 and level>1:
            current_temp = temp
            for _ in range(max_iter):
                new_pos = swap_positions(pos, next_level_nodes)
                current_distance = total_distance(pos, next_level_nodes, tree)
                new_distance = total_distance(new_pos, next_level_nodes, tree)

                if new_distance < current_distance or math.exp((current_distance - new_distance) / current_temp) > random.random():
                    pos = new_pos

                current_temp *= cooling_rate

        level += 1
        current_level_nodes = next_level_nodes

    return pos

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




def radial_tree_layout2(graph, root, radius=3, max_iter=5000, temp=500, cooling_rate=0.2, perturb_angle=0.4, edge_penalty=7.0):

    pos = {}
    visited = set()
    
    def dfs(node, depth, angle_range):
        visited.add(node)
        n_children = len([n for n in graph.neighbors(node) if n not in visited])
        angle_step = (angle_range[1] - angle_range[0]) / max(n_children, 1)

        angle = angle_range[0]
        for child in graph.neighbors(node):
            if child not in visited:
                pos[child] = (depth * radius * np.cos(angle), depth * radius * np.sin(angle))
                dfs(child, depth + 1, (angle - angle_step / 2, angle + angle_step / 2))
                angle += angle_step

    pos[root] = (0, 0)
    dfs(root, 1, (0, 2 * np.pi))
    def count_intersections(pos):
        count = 0
        distance_cost = 0
        for u, v in graph.edges:
            for p, q in graph.edges:
                if u != p and u != q and v != p and v != q:
                    count += int(intersect(pos[u], pos[v], pos[p], pos[q]))
            # Calculate distance from child to predecessor
            distance_cost += np.sqrt((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)
            
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            for i, u in enumerate(neighbors):
                for v in neighbors[i+1:]:
                    # Calculate distance between nodes in the same ring
                    distance_cost -= np.sqrt((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)
                    
        total_cost = count * edge_penalty + distance_cost
        return total_cost




    def intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def swap_positions(pos, nodes):
        new_pos = pos.copy()

        # Find the node with the most intersections
        node_intersections = {node: 0 for node in nodes}
        for u, v in graph.edges:
            for p, q in graph.edges:
                if u != p and u != q and v != p and v != q:
                    if intersect(pos[u], pos[v], pos[p], pos[q]):
                        node_intersections[u] += 1
                        node_intersections[v] += 1
                        node_intersections[p] += 1
                        node_intersections[q] += 1
        node = max(node_intersections, key=node_intersections.get)

        level_radius = np.sqrt(new_pos[node][0] ** 2 + new_pos[node][1] ** 2)
        theta = math.atan2(new_pos[node][1], new_pos[node][0])
        delta_theta = random.uniform(-perturb_angle, perturb_angle)
        theta += delta_theta
        new_pos[node] = (level_radius * np.cos(theta), level_radius * np.sin(theta))
        
        return new_pos

    for _ in range(max_iter):
        current_cost = count_intersections(pos)
        if current_cost == 0:
            break

        intersecting_nodes = set()
        for u, v in graph.edges:
            for p, q in graph.edges:
                if u != p and u != q and v != p and v != q:
                    if intersect(pos[u], pos[v], pos[p], pos[q]):
                        intersecting_nodes.add(u)
                        intersecting_nodes.add(v)
                        intersecting_nodes.add(p)
                        intersecting_nodes.add(q)

        new_pos = swap_positions(pos, intersecting_nodes)
        new_cost = count_intersections(new_pos)

        if new_cost < current_cost or math.exp((current_cost - new_cost) / (temp + 1e-9)) > random.random():
            pos = new_pos

        temp *= cooling_rate

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


def draw_tree(G, tree, root, pos, ax):
    node_colors = color_nodes_based_on_ring(tree, root)

    G_edges_no_selfloop = [(u, v) for u, v in G.edges if u != v]
    tree_edges_no_selfloop = [(u, v) for u, v in tree.edges if u != v]

    nx.draw_networkx_edges(G, pos, edgelist=G_edges_no_selfloop, alpha=0.4, edge_color='gray', arrowstyle='-', ax=ax)
    nx.draw_networkx_edges(tree, pos, edgelist=tree_edges_no_selfloop, alpha=1, edge_color='black', width=2, arrowstyle='-', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=node_colors.keys(), node_color=list(node_colors.values()), ax=ax)
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")


import matplotlib.animation as animation


def interpolate_positions(pos1, pos2, alpha):
    pos = {}
    for node in pos1:
        pos[node] = (pos1[node][0] * (1 - alpha) + pos2[node][0] * alpha,
                     pos1[node][1] * (1 - alpha) + pos2[node][1] * alpha)
    return pos



# ... (rest of the code remains the same)

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


    roots = [node for node in G.nodes]
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
                #print('---------------------------------')
                #print('Loaded root', root)
                #print(pos)
        else:
            #pos = circular_layout(root, tree, radius=1.5, max_iter=400000,
            #                      temp=200, cooling_rate=0.9, perturb_angle=0.5,
            #                      intra_ring_penalty=50, cross_penalty=100)
            pos = radial_tree_layout(tree, root)
            #print('Done root', root)
            with open(pos_file, "wb") as f:
                pickle.dump(pos, f)
        pos_list.append(pos)


    # Calculate the maximum distance from any root
    max_distance = 0
    for tree, root in zip(trees, roots):
        max_distance = max(max_distance, max(nx.shortest_path_length(tree, source=root, target=node) for node in tree.nodes))

    #fig, (ax, cax, pax) = plt.subplots(ncols=3, figsize=(21, 8), gridspec_kw={'width_ratios': [15, 2, 15], 'wspace': 0.5})

    cmap = cm.get_cmap('viridis')
    fig = plt.figure(figsize=(24, 12))
    #gs = GridSpec(nrows=1, ncols=4, width_ratios=[15, 1, 3, 15],wspace=0, hspace=0)
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

        nx.draw_networkx_edges(G, pos, edgelist=G_edges_no_selfloop, alpha=non_tree_edge_alpha, edge_color='black', width=non_tree_edge_width, arrowstyle='-', ax=ax)
        nx.draw_networkx_edges(tree, pos, edgelist=tree_edges_no_selfloop, alpha=tree_edge_alpha, edge_color='black', width=tree_edge_width, arrowstyle='-', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=node_colors1.keys(), node_color=list(node_colors1.values()), node_size=200,linewidths=1, ax=ax)

        ax.axis("off")
        center_circle = patches.Circle((0, 0), radius=0.1, color='red')
        #ax.add_artist(center_circle)
        layers = [0, 1, 2, 3, 4, 5, 6,7]
        for i in range(len(layers)):
            c = plt.Circle((0, 0), i*3, linestyle = '--',color='k', fill=False, alpha = 0.1, clip_on=False)
            ax.add_artist(c)

        ax.set_xlim(-24, 24)
        ax.set_ylim(-24, 24)
        ax.set_aspect('equal', adjustable='box')

        displayed_roots = roots[:transition+1]
        degrees = [degree[root] for root in displayed_roots]
        inv_closenesses = [1/closeness[root] for root in displayed_roots]
        # Find mean inverse closeness for each degree

        stats_dict = pb.aggregate_dict(np.asarray(degrees), np.asarray(inv_closenesses))
        ks, inv_c_mean, errs, stds, counts   = pb.unpack_stat_dict(stats_dict)
        
        pax.errorbar(ks, inv_c_mean, yerr=errs, fmt='x', color='black', capsize=4, capthick=2, elinewidth=2, markersize=7, markeredgewidth=2, alpha = 0.8, label = 'Mean at Degree')
        pax2.axis('off')
        pax1.text(0.25, 0.25, r'$\dfrac{1}{C_r}=-\dfrac{1}{ln(\bar{z})}ln(k_r) +\beta (\bar{z},N)$', transform=pax1.transAxes, fontsize=25, verticalalignment='bottom')
        pax1.axis('off')
        if len(ks) >= 2:
            popt, pcov = sp.optimize.curve_fit(pb.Tim , degrees, inv_closenesses, p0=[1, 1])
            all_ks = np.linspace(min_degree, max_degree, 100)
            pax.plot(all_ks, pb.Tim(all_ks, *popt), 'k--', alpha = 0.8, label = 'Fit')
            pax2.clear()
            pax2.axis('off')
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

            text = r'$\beta = $' + f'{beta:.2f}' + r'$ \pm $' + f'{beta_err:.2f}' + r'$ \quad \quad \bar{z} = $' + f'{z:.2f}' + r'$ \pm $' + f'{z_err:.2f}'
            pax2.text(0.05, 0.2, text , transform=pax2.transAxes, fontsize=25, verticalalignment='bottom')

        pax.scatter(degrees[:-1], inv_closenesses[:-1], color='blue', alpha = 0.5, label = 'Prior Nodes')
        pax.scatter(degrees[-1], inv_closenesses[-1], color='red', alpha = 0.5, label = 'Current Node')
        pax.set_xlabel('Degree of Root, '+ r'$k_r$', fontsize=20)
        pax.set_ylabel('Inverse Closeness ' +  r'$1/C$', fontsize=20)
        pax.set_xlim(min_degree-0.1, max_degree+0.1)
        pax.set_ylim(1/max_closeness - 0.1, 1/min_closeness+0.1)
        pax.set_xscale('log')
        pax.legend(loc='upper right', fontsize=20)
        pax.grid(True)

    ani = animation.FuncAnimation(fig, animate, frames=(len(roots) - 1) * (frames_per_transition + pause_frames) + pause_frames, interval=15, blit=False, repeat=False)
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0)
    writer_video = FFMpegWriter(fps=25, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('animation.mp4', writer=writer_video)
    #plt.show()



    end = time.time()
    print("Time taken to process: ", end - start)

if __name__ == "__main__":
    main()
