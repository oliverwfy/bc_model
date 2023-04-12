from model import simulate_bc_own_belief_malicious_1_total_variation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

alpha = 0.1
init_x = None
mal_x = 0.2
malicious = 0.1
threshold = 0.3


result = simulate_bc_own_belief_malicious_1_total_variation(simulation_times=100, pop_n=10, max_iteration=80, k=3, init_x = init_x,
                                                            mal_x = mal_x,alpha=alpha, prob_evidence=0.1, malicious=malicious, threshold= threshold,
                                                            noise=None, pooling=True, dampening = False)


confidence_mat = result['confidence_mat']
avg_belief = result['belief_avg_true_good']
np.save('confidence_mat.npy', confidence_mat)
np.save('avg_belief.npy', avg_belief)
#
#
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
# step_ls = [0, 5, 10, 15, 20, 30, 40, 50]
# for step in step_ls:
#     plt.figure()
#     G = nx.DiGraph()
#
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
#     plt.title(f'time step = {step}',  fontsize=14)
#     nx.draw(G, with_labels=False, node_color=color_map, edge_color = 'gray', width=2.0, pos=pos)
#     plt.savefig(f'bc_total_variation/' + f"influence_network_step_{step}.png")
#
