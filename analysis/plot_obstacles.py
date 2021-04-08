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

print("starting")

elev = 0  # 20
azim = 90  # 95

SEEDS = [2, 4, 7, 8, 22, 25, 36, 39, 49]  # range(9300, 9351)
ROT = [0, 1, 0, 0, 1, 0, 2, 1, 1]

fig = plt.figure(figsize=(7, 9))

ls = LightSource(0, 90)

n = 0
for b in SEEDS:

    with open("obstacles_data/body_data_seed_{}.pickle".format(b), 'rb') as handle:
        fit, unturned_bot = pickle.load(handle)

    bot = np.rot90(unturned_bot, k=ROT[n], axes=(1, 0))

    while True:  # shift down until in contact with surface plane
        if np.sum(bot[:, :, 0]) == 0:
            bot[:, :, :-1] = bot[:, :, 1:]
            bot[:, :, -1] = np.zeros_like(bot[:, :, -1])
        else:
            break

    bot = np.pad(bot, pad_width=((bot.shape[0],)*2, (bot.shape[0],)*2, (0,)*2), mode='constant', constant_values=0)

    world = np.zeros_like(bot)
    for i in range(0, world.shape[0], 5):
        for j in range(0, world.shape[1], 5):
            world[i, j, 0] = 1

    print("printing bot {}".format(n))
    n += 1
    ax = fig.add_subplot(4, 3, n, projection='3d')
    ax.set_xlim([0, bot.shape[0]])
    ax.set_ylim([0, bot.shape[0]])
    ax.set_zlim([0, bot.shape[0]])

    # ax.set_aspect('equal')
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    b = np.pad(bot, pad_width=1, mode='constant', constant_values=0)
    neigh = np.concatenate((
        b[2:, 1:-1, 1:-1, None], b[:-2, 1:-1, 1:-1, None],
        b[1:-1, 2:, 1:-1, None], b[1:-1, :-2, 1:-1, None],
        b[1:-1, 1:-1, 2:, None], b[1:-1, 1:-1, :-2, None]), axis=3)
    bot_locations = np.where(np.sum(neigh, axis=3) > 0)
    world[bot_locations] = 0

    x, y, z = np.indices((3*bot.shape[0],)*3)
    ax.voxels(world, facecolors=(0, 0, 0), edgecolor='k', linewidth=0.1, shade=True, lightsource=ls, alpha=0.6)
    ax.voxels(bot, facecolors=(1, 0, 1), edgecolor='k', linewidth=0.1, shade=True, lightsource=ls, alpha=0.6)
    adj_fit = fit-1
    ax.text(0.5*bot.shape[0], 1*bot.shape[1], -1*bot.shape[2], "FG: {}".format(round(adj_fit, 2)), fontsize=12, ha='center')

print("saving!")
# fig.subplots_adjust(wspace=-0.25, hspace=-0.05)
fig.subplots_adjust(wspace=-0.2, hspace=-0.2)
bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

plt.savefig("ObstacleChampsLoRes{}.png".format(elev), bbox_inches='tight', dpi=300, transparent=True)
plt.savefig("ObstacleChampsHiRes{}.png".format(elev), bbox_inches='tight', dpi=800, transparent=True)

