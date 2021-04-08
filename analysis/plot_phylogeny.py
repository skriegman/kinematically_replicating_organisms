import numpy as np
import sys
# import cPickle
import pickle
import subprocess as sub
from glob import glob

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.patches as mpatches
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['legend.frameon'] = 'True'
matplotlib.rcParams["legend.framealpha"] = 0.75
matplotlib.rcParams["legend.fancybox"] = True

sns.set(color_codes=True, context="poster")
sns.set_style("white")

# sns.set_palette("hls", 8)

RUN = 9324
# print RUN

with open("rainbow_data/phylogenetic_data_seed_{}.pickle".format(RUN), 'rb') as handle:
    # data = cPickle.load(handle)
    data = pickle.load(handle)

a_gen = np.array(data["gen"])
a_age = np.array(data["age"])
a_fit = np.array(data["fit"])
a_cid = np.array(data["id"])
a_pid = np.array(data["pid"])
# a_body = np.array(data["body"])

def get_ancestors(twig, ancestors):
    pid = a_pid[np.where(a_cid == twig)[0][0]]
    if pid > -1:
        ancestors += [pid]
        get_ancestors(pid, ancestors)
    return ancestors

def get_descendants(root, descendants):
    cids = a_cid[np.where(a_pid == root)[0]]
    fits = a_fit[np.where(a_pid == root)[0]]

    for cid, fit in zip(cids, fits):
        if (root, cid, fit) not in descendants:
            descendants += [(root, cid, fit)]
            # print descendants
            get_descendants(cid, descendants)
    return descendants

champ = a_cid[np.argmax(a_fit)]
ancestors = [champ]
ancestors = get_ancestors(champ, ancestors)

start = ancestors[-1]
descendants = []
descendants = get_descendants(start, descendants)

# print(ancestors)
# print(descendants)

nodes_in_tree = []

G = nx.DiGraph()

fit = a_fit[np.where(a_cid == start)[0][0]]
G.add_node(start, fit=fit)

for branch in descendants:
    pid, cid, fit = branch
    G.add_node(cid, fit=fit)
    G.add_edge(pid, cid)

# print "finally, drawing it..."

node_attrs = nx.get_node_attributes(G, 'fit')
unranked_attr = []
for node, attr in node_attrs.items():
    unranked_attr += [attr]

array = np.array(unranked_attr)
order = array.argsort()
ranks = order.argsort()
color_map = [cm.cool(a) for a in ranks/float(len(unranked_attr)-1)]

color_map = [cm.cool(a) for a in (array-np.min(array))/(np.max(array)-np.min(array))]


# draw tree
plt.figure(figsize=(8, 4))
# pos=graphviz_layout(G, prog='dot')
pos=graphviz_layout(G, prog='dot', args="-Grankdir=LR")
nx.draw(G, pos, alpha=1, with_labels=False, arrows=True, arrowsize=15, arrowstyle='-|>',
        linewidths=2, edgecolors="black", width=2, edge_color="black", node_color=color_map, node_size=400)

if RUN == 9324:
    label_dict = {start: "b", 466: "c", 595: "d"}
    label_pos = {}
    for node, coords in pos.items():
        if node == 466:
            label_pos[node] = (coords[0]-0.25, coords[1]-1)
        elif node == 595:
            label_pos[node] = (coords[0]+0.5, coords[1]-1)
        else:
            label_pos[node] = (coords[0], coords[1]-1)

    nx.draw_networkx_labels(G, label_pos, label_dict)

pos_attrs = {}
for node, coords in pos.items():
    pos_attrs[node] = (coords[0], coords[1] - 12*2.15)

# node_attrs = nx.get_node_attributes(G, 'fit')
custom_node_attrs = {}
for node, attr in node_attrs.items():
    custom_node_attrs[node] = "FG: {}".format(round(attr,2))

nx.draw_networkx_labels(G, pos_attrs, labels=custom_node_attrs, font_size=10)

xmax = 1.1 * max(xx for xx, yy in pos.values())
ymax = 1.1 * max(yy for xx, yy in pos.values())
plt.xlim(-3, xmax)
plt.ylim(-20, ymax)


plt.savefig("phylogeny_{}.png".format(RUN), bbox_inches='tight', dpi=600, transparent=True)

