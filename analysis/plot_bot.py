import random
import numpy as np
import sys
from time import time
import pickle
import subprocess as sub
from glob import glob
import colorsys
import re

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
color_palette = [(1, 0, 1, 0.8), (0, 1, 0, 0.8)]
# cmap = cm.get_cmap('rainbow')
cmap = cm.get_cmap('cool')

# print cmap(1.0)

with open("rainbow_data/fitness_data.pickle", 'rb') as handle:
    fits = pickle.load(handle)

print("starting")

SEEDS = range(49)  # skip 50th

fig = plt.figure(figsize=(7, 8))


ls = LightSource(0, 90)

n = 0
for b in SEEDS:

    with open("rainbow_data/body_data_seed_{}.pickle".format(b), 'rb') as handle:
        bot = pickle.load(handle)

    print("printing bot {}".format(n))
    n += 1
    ax = fig.add_subplot(8, 7, n, projection='3d')
    ax.set_xlim([0, bot.shape[0]])
    ax.set_ylim([0, bot.shape[0]])
    ax.set_zlim([0, bot.shape[0]])

    # ax.set_aspect('equal')
    ax.view_init(elev=80, azim=100)
    ax.set_axis_off()

    x, y, z = np.indices((7, 7, 5))
    ax.voxels(bot, facecolors=(1, 0, 1), edgecolor='k', linewidth=0.1, shade=True, lightsource=ls, alpha=0.6)
    ax.text(0.5 * bot.shape[0], 1.0*bot.shape[1], -6*bot.shape[2], "FG: {}".format(round(fits[n-1]-1, 2)), fontsize=10, ha='center')

# fig.subplots_adjust(wspace=-0.25, hspace=-0.05)
fig.subplots_adjust(wspace=-0.05, hspace=-0.05)
bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig("RunChamps.png", bbox_inches='tight', dpi=800, transparent=True)


