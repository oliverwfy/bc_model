import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# def convert_to_hex(rgba_color) :
#     red = int(rgba_color[0]*255)
#     green = int(rgba_color[1]*255)
#     blue = int(rgba_color[2]*255)
#     return '#%02x%02x%02x' % (red, green, blue)
#
#
# def confidence_mat_parser(confidence_mat, step=0, threshold=0.5):
#     edge = []
#     mat = confidence_mat[step]
#     for i in range(mat.shape[0]):
#         for j in range(mat.shape[1]):
#             if mat[i,j] < threshold:
#                 edge.append((i,j))
#     return edge
#
#
# confidence_mat = np.round(np.load('confidence_mat.npy').mean(axis=0),6)
#
#
# cmap = plt.get_cmap('Greens')
#
# step_ls = [0, 10, 20, 30, 40, 50, 60, 70, 80]
# for step in step_ls:
#     plt.figure()
#     G = nx.DiGraph()
#
#     threshold = 0.3
#     G.add_nodes_from([i for i in range(10)])
#     G.add_edges_from(confidence_mat_parser(confidence_mat,step=step, threshold=threshold))
#
#
#     pos = nx.circular_layout(G)
#
#     color_map = []
#     for node in G:
#         if node < 1:
#             color_map.append(convert_to_hex(plt.get_cmap('Reds')(150)))
#         else:
#             rgba = cmap(150)
#             color_map.append(convert_to_hex(rgba))
#
#     plt.title(f'time step = {step}',  fontsize=15)
#     nx.draw(G, with_labels=False, node_color=color_map, edge_color = 'gray', width=2.0, pos=pos)
#     plt.savefig(f'bc_total_variation/' + f"influence_network_step_{step}.png")
#

def confidence_interval(data):
    n = data.shape[1]
    std = data.std(axis=1)
    return 1.960*std/np.sqrt(n)

avg_belief = np.load('avg_belief.npy')
ci = confidence_interval(avg_belief)
plt.plot(range(len(avg_belief)), avg_belief.mean(axis=1), color='green')
plt.axhline(y=0.95, color='gray', linestyle='--')
plt.fill_between(range(len(avg_belief)), (avg_belief.mean(axis=1)-ci), (avg_belief.mean(axis=1)+ci), alpha=0.2)

plt.xlabel('time step', fontsize=14)
plt.ylabel('average belief', fontsize=14)
plt.title(r'$\gamma = 0.3,\alpha = 0.1, \beta = 0.1, \epsilon=0$', fontsize=14)
plt.legend(['avg belief (95% confidence interval)', 'consensus condition'],fontsize=14)
plt.savefig(f'bc_total_variation/' + f'avg_belief_normal.png')