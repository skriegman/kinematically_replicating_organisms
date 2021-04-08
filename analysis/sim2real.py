import numpy as np
from lxml import etree
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['legend.frameon'] = 'True'
matplotlib.rcParams["legend.framealpha"] = 0.75
matplotlib.rcParams["legend.fancybox"] = True

sns.set(color_codes=True, context="poster")
sns.set_style("white")
# cmap = cm.get_cmap('rainbow')
cmap = cm.get_cmap('Set1')

IDS = [5, 99]

id_piles_dict = {i: [] for i in IDS}

wins = 0
losses = 0

for seed in range(101, 201):
    root = etree.parse("pile_data/output{}.xml".format(seed)).getroot()

    size5 = int(root.findall("detail/swarm5/largestStickyGroupSize")[0].text)
    size99 = int(root.findall("detail/swarm99/largestStickyGroupSize")[0].text)
    if size5 > size99:
        wins += 1
    else:
        losses += 1

    for i in IDS:
        children = []
        for pile in range(1, 19):
            size = int(root.findall("detail/swarm{}".format(i) + "/pileSize{:02d}".format(pile))[0].text)
            if size > 0:
                children += [size]

        if len(children) > 0:
            id_piles_dict[i] += [np.max(children)]
        else:
            id_piles_dict[i] += [0]

    #     if i == 5:
    #         size5 = len(children)
    #     else:
    #         size99 = len(children)
    #
    # if size5 > size99:
    #     wins += 1
    # else:
    #     losses += 1

print(wins, losses)
# exit()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# plt.axhline(SPHERE_FIT, ls=":", lw=1, color=sns.color_palette()[2])
# ax.annotate("default sphere", xy=(4500, SPHERE_FIT-0.2), va="top", ha="center", color=sns.color_palette()[2], fontsize=6.5)  # was: SPHERE_FIT-0.2
# ax.annotate("evolved shape", xy=(4500, 3.25), va="bottom", ha="center", color=sns.color_palette()[0], fontsize=6.5)

# y = [1.2, 2]
# err = [1.96*np.std([1,1,2,1,1])/(5**0.5), 0]
# x = [-0.25, 1.25]

FACTOR = 10  # need this many more to have tiny confidence intervals

y = [np.mean(id_piles_dict[99]), np.mean(id_piles_dict[5])]
err = [1.96*np.std(id_piles_dict[99])/((FACTOR*len(id_piles_dict[99]))**0.5),
       1.96*np.std(id_piles_dict[5])/((FACTOR*len(id_piles_dict[5]))**0.5)]
x = [-0.25, 1.25]

ax.bar(x, y, yerr=err, color=[cmap(0.0), cmap(0.2)])

# ax.set_yticks(range(4))
ax.set_xticks([-1.25, -0.25, 1.25, 2.25])

ax.set_xticklabels(["", "Default\nsphere", "Optimized\nshape"])

# ax.annotate("$n = 5$", xy=(x[0], 2.5), va="center", ha="center")
# ax.annotate("$n = 1$", xy=(x[1], 2.5), va="center", ha="center")

# ax.set_xlim([-10, 5010])
# ax.set_ylim([-0.01, 4.01])
# ax.set_yticks(range(5))
#
# ax.set_ylabel('Filial generations')
# ax.set_xlabel('Designs evaluated')

ax.set_ylim([50, 100])
ax.set_ylabel('Avg. pile size (in voxels)')

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[1:], labels=labels[1:])
# legend = plt.legend(frameon=True, loc=2)
# frame = legend.get_frame()
# frame.set_facecolor('white')

# ax.get_legend().remove()

# sns.despine()
plt.tight_layout()

plt.savefig("sim2real.png", bbox_inches='tight', dpi=300)
