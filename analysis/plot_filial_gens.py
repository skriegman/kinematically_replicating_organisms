import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
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

lables = {"1": 2.81, "0.5": 1.51, "0": 0.74}

for NUM in [1, 0.5, 0]:

    G = nx.DiGraph()

    for n in range(9):
        G.add_node(n)

    if NUM == 0.5:
        G.add_node(11)
        for n in [1, 2]:
            G.add_edge(n, 11)

    if NUM == 1:
        for n in range(9, 12):
            G.add_node(n)

        for n in [1, 2, 3, 5, 6]:
            G.add_edge(n, 9)

        for n in [6, 7]:
            G.add_edge(n, 10)

        for n in [1]:
            G.add_edge(n, 11)

        G.add_node(12)
        G.add_edge(11, 12)


    # draw tree
    plt.figure(figsize=(4, 4))
    if NUM == 1:
        pos=graphviz_layout(G, prog='dot', args="-Grankdir=LR")

    color = NUM
    if NUM == 0.5:
        color = (1.51-0.69)/(2.81-0.69)

    nx.draw(G, pos, alpha=1, with_labels=False, arrows=True, arrowsize=15, arrowstyle='->',
            linewidths=2, edgecolors="black", width=2, edge_color="black", node_color=cm.cool(float(color)), node_size=400)

    xmax = 1.05 * max(xx for xx, yy in pos.values())
    ymax = 1.05 * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    # plt.annotate("FG: {}".format(lables[str(NUM)]), (0.65*xmax, 0.01*ymax), fontsize=24, color="black")

    plt.savefig("filial_tree_{}.png".format(NUM), bbox_inches='tight', dpi=600, transparent=True)

