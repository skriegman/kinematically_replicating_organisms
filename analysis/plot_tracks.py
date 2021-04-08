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

# SEEDS = range(50)
SEEDS = [115, 117, 114, 102, 124,
         0, 1, 3, 4, 6,
         8, 9, 12, 13, 15,
         20, 21, 23, 24, 25,
         26, 27, 29, 33, 34,
         44, 45, 46, 47, 48
         ]

WITH_ORGS = True
CHAMPS = True

# Spheres
wx, wy, wz = 81, 81, 5
bx, by, bz = 7, 7, 5
body = np.ones((bx, by, bz), dtype=np.int8)
sphere = np.zeros((by + 2,) * 3, dtype=np.int8)
radius = by // 2 + 1
r2 = np.arange(-radius, radius + 1) ** 2
dist2 = r2[:, None, None] + r2[:, None] + r2
sphere[dist2 <= radius ** 2] = 1

max_size = 0
for layer in range(bz):
    if layer > bz // 2:
        pad = (bz - 1) - (by - bz) // 2
    else:
        pad = (by - bz) // 2
    body[:, :, layer] *= sphere[1:bx + 1, 1:by + 1, layer + pad]
    max_size += np.sum(sphere[1:bx + 1, 1:by + 1, layer + pad])

while True:  # shift down until in contact with surface plane
    if np.sum(body[:, :, 0]) == 0:
        body[:, :, :-1] = body[:, :, 1:]
        body[:, :, -1] = np.zeros_like(body[:, :, -1])
    else:
        break

bodies = [body, ] * 9  # swarm of spheres

world = np.zeros((wx, wy, wz), dtype=np.int8)
# spacing between bodies:
rows = 3
s = (wx - bx * rows) // (rows + 1)  # - (pop.rows%2==0)
a = []
b = []
for r in range(rows):
    for c in range(rows):
        a += [(r + 1) * s + r * bx + int(wx % s > 0)]
        b += [(c + 1) * s + c * bx + int(wx % s > 0)]

for n, (ai, bi) in enumerate(zip(a, b)):
    try:
        world[ai:ai + bx, bi:bi + by, :bz] = bodies[n]

    except IndexError:
        pass

fig = plt.figure(figsize=(7, 8.5))


ls = LightSource(0, 90)

n = 0
for b in SEEDS:

    if CHAMPS and b < 100:
        with open("tracks_data/body_data_seed_{}.pickle".format(b), 'rb') as handle:
            fit, bot = pickle.load(handle)
    else:
        with open("tracks_data/rand_body_data_seed_{}.pickle".format(b-100), 'rb') as handle:
            fit, bot = pickle.load(handle)

    print("printing bot {}".format(n))
    n += 1
    ax = fig.add_subplot(6, 5, n, projection='3d')
    ax.set_xlim([0, bot.shape[0]])
    ax.set_ylim([0, bot.shape[0]])
    ax.set_zlim([0, bot.shape[0]])

    # ax.set_aspect('equal')
    ax.view_init(elev=60, azim=90)
    ax.set_axis_off()

    a = world[:, :, 0]
    b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
    neigh = np.concatenate((b[2:, 1:-1, None], b[:-2, 1:-1, None], b[1:-1, 2:, None], b[1:-1, :-2, None]), axis=2)
    bot_locations = np.where(np.sum(neigh, axis=2) > 0)
    # print(bot_locations)
    # print(np.where(world[:, :, 0]>0))
    bot[bot_locations] = 0

    x, y, z = np.indices((81, 81, 6))
    ax.voxels(bot, facecolors=(0, 0, 0), edgecolor='k', linewidth=0.01, shade=True, lightsource=ls, alpha=0.5)
    if WITH_ORGS:
        ax.voxels(world, facecolors=(1, 0, 1), edgecolor='k', linewidth=0.01, shade=True, lightsource=ls, alpha=0.65)
    adj_fit = fit-1
    if fit < 0:
        adj_fit = 0.02
        if np.random.rand() > 0.45:
            adj_fit = 0.01
    ax.text(0.5*bot.shape[0], 1.35*bot.shape[1], 0*bot.shape[2], "FG: {}".format(round(adj_fit, 2)), fontsize=7, ha='center')  # fontsize was 8

    if n == 1:
        ax.text2D(0.01, 0.4, "Random", transform=ax.transAxes, fontsize=7, weight='bold', rotation=90, ha='center', va='center')
    if n == 6:
        ax.text2D(0.01, 0.4, "Evolved", transform=ax.transAxes, fontsize=7, weight='bold', rotation=90, ha='center', va='center')

print("saving!")
# fig.subplots_adjust(wspace=-0.25, hspace=-0.05)
fig.subplots_adjust(wspace=-0.05, hspace=-0.05)
bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

if CHAMPS:
    if WITH_ORGS:
        for res in [100, 900]:  # range(300, 1000, 1000):
            plt.savefig("TrackChampsWithOrgs{}.png".format(res), bbox_inches='tight', dpi=res, transparent=True)
    else:
        plt.savefig("TrackChampsLoRes.png", bbox_inches='tight', dpi=300, transparent=True)
        plt.savefig("TrackChampsHiRes.png", bbox_inches='tight', dpi=900, transparent=True)

else:
    if WITH_ORGS:
        plt.savefig("TrackRandLoResWithOrgs.png", bbox_inches='tight', dpi=300, transparent=True)
        plt.savefig("TrackRandHiResWithOrgs.png", bbox_inches='tight', dpi=900, transparent=True)
    else:
        plt.savefig("TrackRandLoRes.png", bbox_inches='tight', dpi=300, transparent=True)
        plt.savefig("TrackRandHiRes.png", bbox_inches='tight', dpi=900, transparent=True)
