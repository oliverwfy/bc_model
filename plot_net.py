import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def convert_to_hex(rgba_color) :
    red = int(rgba_color[0]*255)
    green = int(rgba_color[1]*255)
    blue = int(rgba_color[2]*255)
    return '#%02x%02x%02x' % (red, green, blue)


def confidence_mat_parser(confidence_mat, step=0, threshold=0.5):
    edge = []
    mat = confidence_mat[step]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j] < threshold:
                edge.append((i,j))
    return edge


confidence_mat = np.round(np.load('confidence_mat.npy').mean(axis=0),6)


cmap = plt.get_cmap('Greens')


G = nx.DiGraph()
step = 49
threshold = 0.2
G.add_nodes_from([i for i in range(10)])
G.add_edges_from(confidence_mat_parser(confidence_mat,step=step, threshold=threshold))


pos = nx.circular_layout(G)

color_map = []
for node in G:
    if node < 1:
        color_map.append(convert_to_hex(plt.get_cmap('Reds')(150)))
    else:
        rgba = cmap(150)
        color_map.append(convert_to_hex(rgba))


nx.draw(G, with_labels=False, node_color=color_map, edge_color = 'gray', width=2.0, pos=pos)
plt.show()

