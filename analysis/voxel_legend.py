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

from matplotlib.colors import LightSource, to_hex
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

ls = LightSource(0, 45)

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])

ax.view_init(elev=20, azim=110)
ax.set_axis_off()

x, y, z = np.indices((11, 11, 11))

light = (x == 9) & (y == 9) & (z == 1)
power = (x == 9) & (y == 9) & (z == 3)
metal = (x == 9) & (y == 9) & (z == 5)
adult = (x == 9) & (y == 9) & (z == 7)
cells = (x == 9) & (y == 9) & (z == 9)

voxels = adult | cells | metal | light | power

colors = np.empty(voxels.shape, dtype=object)
colors[adult] = to_hex((1, 0, 1))
colors[cells] = to_hex((0, 1, 0))
colors[metal] = to_hex((0, 0, 1))
colors[light] = to_hex((0.75, 1, 1))
colors[power] = to_hex((1, 1, 0))

ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=0.5, shade=False, lightsource=ls, alpha=1)
ax.text(8.9, 8, 9, 'Stem cells', fontsize=12, ha='left', va='center')
ax.text(8.9, 8, 7, 'Mature ciliated cells', fontsize=12, ha='left', va='center')
ax.text(8.9, 8, 5, 'Conductor', fontsize=12, ha='left', va='center')
ax.text(8.9, 8, 3, 'Current', fontsize=12, ha='left', va='center')
ax.text(8.9, 8, 1, 'Resistor ("light bulb")', fontsize=12, ha='left', va='center')

# fig.subplots_adjust(wspace=-0.05, hspace=-0.05)
bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig("Legend.png", bbox_inches='tight', dpi=300, transparent=True)

