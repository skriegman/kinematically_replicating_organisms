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

from matplotlib.colors import LightSource
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
cmap = cm.get_cmap('cool')


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
a_body = np.array(data["body"])

def get_ancestors(twig, ancestors):
    pid = a_pid[np.where(a_cid == twig)[0][0]]
    if pid > -1:
        ancestors += [pid]
        get_ancestors(pid, ancestors)
    return ancestors

def get_descendants(root, descendants, d_bods):
    cids = a_cid[np.where(a_pid == root)[0]]
    fits = a_fit[np.where(a_pid == root)[0]]
    bodies = a_body[np.where(a_pid == root)[0]]

    for cid, fit, body in zip(cids, fits, bodies):
        # print(body.shape)
        if (root, cid, fit) not in descendants:
            descendants += [(root, cid, fit)]
            d_bods += [body]
            # print descendants
            get_descendants(cid, descendants, d_bods)
    return descendants, d_bods

champ = a_cid[np.argmax(a_fit)]
ancestors = [champ]
ancestors = get_ancestors(champ, ancestors)

start = ancestors[-1]
descendants = []
d_bods = []
descendants, d_bods = get_descendants(start, descendants, d_bods)


unranked_fits = []
for n in range(len(descendants)):
    pid, cid, fit = descendants[n]
    unranked_fits += [fit]

array = np.array(unranked_fits)
order = array.argsort()
ranks = order.argsort()
color_map = [cm.cool(a) for a in (array-np.min(array))/(np.max(array)-np.min(array))]


ls = LightSource(0, 90)
r = 0
for n in range(18):#[1, 13, 6]:
    fig = plt.figure(figsize=(7, 8))

    pid, cid, fit = descendants[n]
    bot = d_bods[n]

    # print(bot.shape)
    print("printing bot {}".format(n))
    r += 1
    ax = fig.add_subplot(8, 7, 1, projection='3d')
    ax.set_xlim([0, bot.shape[0]])
    ax.set_ylim([0, bot.shape[0]])
    ax.set_zlim([0, bot.shape[0]])

    ax.view_init(elev=90, azim=90)
    ax.set_axis_off()

    x, y, z = np.indices((7, 7, 5))
    ax.voxels(bot, facecolors=color_map[n], edgecolor='k', linewidth=0.1, shade=True, lightsource=ls, alpha=0.6)
    ax.text(0.5*bot.shape[0], 1.55*bot.shape[1], 0.5*bot.shape[2], "FG: {}".format(round(fit, 2)), fontsize=12, ha='center')

    fig.subplots_adjust(wspace=-0.05, hspace=-0.05)
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig("LineageBody{}.png".format(n), bbox_inches='tight', dpi=600, transparent=True)


