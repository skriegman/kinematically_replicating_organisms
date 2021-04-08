import numpy as np
import pickle

import seaborn as sns
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import LightSource
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


RUN = 9300
# print RUN


# with open("rainbow_data/body_data_seed_{}.pickle".format(RUN-9300), 'rb') as handle:
#     champ = pickle.load(handle)
#
# champ += np.rot90(champ)


BODY_SIZE = (25, 25, 1)

l, w, h = BODY_SIZE

body = np.ones(BODY_SIZE, dtype=np.int8)

sphere = np.zeros((w+2,)*3, dtype=np.int8)
radius = w//2+1
r2 = np.arange(-radius, radius+1)**2
dist2 = r2[:, None, None] + r2[:, None] + r2
sphere[dist2 <= radius**2] = 1

for layer in range(h):
    if False: #layer >= h//2:
        pad = 12+4+4
    else:
        pad = 4
    body[:, :, layer] *= sphere[1:l+1, 1:w+1, layer+pad]

champ = body

print("printing bot")

ls = LightSource(0, 90)

fig = plt.figure(figsize=(7, 8))

ax = fig.add_subplot(8, 7, 1, projection='3d')
ax.set_xlim([0, champ.shape[0]])
ax.set_ylim([0, champ.shape[0]])
ax.set_zlim([0, champ.shape[0]])

ax.view_init(elev=10, azim=110)
ax.set_axis_off()

x, y, z = np.indices(BODY_SIZE)
ax.voxels(champ, facecolors=(1, 0, 1), edgecolor='k', linewidth=0.1, shade=True, lightsource=ls, alpha=0.8)

fig.subplots_adjust(wspace=-0.05, hspace=-0.05)
bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig("Champ{}.png".format(RUN), bbox_inches='tight', dpi=3000, transparent=True)


