import numpy as np
from statsmodels.stats import multitest
from scipy.stats import mannwhitneyu, binom_test, wilcoxon, ranksums, pearsonr, spearmanr

x1_s25 = [0.150529,	0.150529, 0.1445211, 0.1271895,	0.1227579,	0.1159079,	0.1118974, 0.1106526, 0.1059184, 0.1005526]

x2_s75 = [0.3148237, 0.3090947, 0.2879395, 0.2832842, 0.2292947, 0.2160474,0.2085421,0.1935237,0.1671842,0.1652658]

x3_s150 = [0.462979, 0.3746105, 0.3114158, 0.3016579, 0.2966579,0.2848421,0.2834289,0.27555, 0.253179 ,0.2522474]

x4_c61 = [0.4521711, 0.4222342, 0.3721605, 0.3390132, 0.3316211, 0.31915, 0.3051737, 0.294971, 0.2779026, 0.2717026]

x5_s2x52 = [0.2709368, 0.2563316, 0.2349632, 0.2306789, 0.2193553, 0.1972816, 0.1907842, 0.1854105, 0.1842105, 0.1824737]

x6_sflat105 = [0.5708763, 0.2848421, 0.2837237, 0.2662974, 0.253071, 0.2336026, 0.2221158, 0.2183421, 0.2158526, 0.210657]

x7_c61 = [0.4383921,0.3548237,0.3314105, 0.3000474, 0.2992605, 0.2852816,0.2686263, 0.26635,0.2570342,0.2542158]

x8_c91 = [0.6386021,0.4772063,0.4711292,0.4204208,0.41675,0.3883958,0.3793729,0.3611813,0.3286396, 0.323275]


s = np.array([np.array(x1_s25)/25.]+[np.array(x2_s75)/75.]+[np.array(x3_s150)/150.]+[np.array(x5_s2x52)/52.]+[np.array(x6_sflat105)/105.])

c = np.array([np.array(x4_c61)/61.]+[np.array(x7_c61)/61.]+[np.array(x8_c91)/91.])

s = s.flatten()
c = c.flatten()

print(len(s), len(c))

print(mannwhitneyu(s,c))

gs = [1, 1, 2, 1, 1]
gc = [3, 2, 4]

print(mannwhitneyu(gs,gc))

print("fdr: ", multitest.fdrcorrection([3.9e-5, 1.2e-7, 0.018817, 1e-7], alpha=0.05, method='indep', is_sorted=False))

means = [0.12404554, 0.2375, 0.30965685, 0.33861, 0.21524263, 0.27593815, 0.30554421, 0.4204973]
gens = [1, 1, 2, 3, 1, 1, 2, 4]

print("pearson: ", pearsonr(means, gens))
print("spearman: ", spearmanr(means, gens))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D


# sns.set_theme(style="white")

# data = np.array([x1_s25, x5_s2x52, x4_c61, x7_c61, x2_s75,  x8_c91, x6_sflat105, x3_s150])
# df = pd.DataFrame(data.T)
# df.columns = ['s25', 's2x52', 'c61', 'cc61', 's75', 'c91', 'sflat105', 's150']

size = np.array(x4_c61 + x7_c61 + x8_c91 + x1_s25 + x5_s2x52 + x2_s75 + x6_sflat105 + x3_s150)
cells = np.array([61.,]*10 + [61.,]*10 + [91.,]*10 + [25.,]*10 + [52.,]*10 + [75.,]*10 + [105.,]*10 + [150.,]*10)
shape = np.array(["semi torus",]*10 + ["semi torus",]*10 + ["semi torus",]*10 + ["sphere",]*10 + ["sphere",]*10 + ["large sphere",]*10 + ["compressed",]*10 + ["sphere",]*10)

df = pd.DataFrame(np.array([size, cells, shape]).T)
df.columns = ['size', 'cells', 'shape']

df["size"] = pd.to_numeric(df["size"], downcast="float")
df["cells"] = pd.to_numeric(df["cells"], downcast="float")

print(df.dtypes)

# Initialize the figure
f, ax = plt.subplots(figsize=(5, 6))
sns.despine(top=True, right=True)

ax.set_xlim([0, 175])

sns.regplot(x=df["cells"][df["shape"] != "semi torus"], y=df["size"][df["shape"] != "semi torus"],
            scatter=False,
            logx=True,
            truncate=False,
            ci=95, color=(.5, .5, .5), line_kws={"ls": "--", "lw": 1, "zorder": 0})

# Show each observation with a scatterplot
palette={"semi torus": (1, 0.0, 1, 1),
         "sphere": (0.2, 0.2, 0.2, 1),
         "large sphere": (.6, .6, .6, 1),
         "compressed": (1, 1, 1, 1)}
sns.scatterplot(x="cells", y="size", hue="shape", data=df, zorder=1,
                # marker=TextPath((0,0), "c"),
                edgecolor=(0, 0, 0, 1),
                s=100,
                palette=palette
                )

# Improve the legend
# legend_elements = [Circle((0,0), facecolor='orange', edgecolor='r', label='asdf')]
handles, labels = ax.get_legend_handles_labels()
new_handles = [Line2D([0], [0], marker='o', markerfacecolor=palette["semi torus"], markeredgecolor='black', markersize=10, ls=''),
               Line2D([0], [0], marker='o', markerfacecolor=palette["sphere"], markeredgecolor='black', markersize=10, ls=''),
               Line2D([0], [0], marker='o', markerfacecolor=palette["large sphere"], markeredgecolor='black', markersize=10, ls=''),
               Line2D([0], [0], marker='o', markerfacecolor=palette["compressed"], markeredgecolor='black', markersize=10, ls=''),
               ]
ax.legend(new_handles, labels, loc=(0.89, 0.77), fontsize=12)
# ax.legend(title=None)

ax.set_xlim([0, 175])
ax.set_xticks(range(0, 176, 25))
ax.set_ylim([0, 0.7])
ax.set_ylabel('Diameter of 10 largest offspring (mm)', fontsize=16)
ax.set_xlabel('Dissociated stem cell density (cells/mm$^2$)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

axins = inset_axes(ax, "100%", "100%", bbox_to_anchor=(1.03, 0.07, .3, .3), bbox_transform=ax.transAxes)

none61 = [0.075905263, 0.071828947, 0.071052632, 0.066418421, 0.064513158, 0.064084211, 0.063157895, 0.061434211, 0.061434211, 0.061378947]
none83 = [0.073165789, 0.071102632, 0.063268421, 0.059136842, 0.057894737, 0.053673684, 0.052697368, 0.049652632, 0.047442105, 0.047442105]
none150 = [0.078242105, 0.074105263, 0.072260526, 0.069526316, 0.067247368, 0.066989474, 0.065789474, 0.059663158, 0.058431579, 0.058431579]

n = np.array(none61+none83+none150)
n /= np.array([61.,]*10 + [83.,]*10 + [150.,]*10)
ratios = np.append(s.copy(), c.copy(), axis=0)
ratios = np.append(n.copy(), ratios.copy(), axis=0) * 1000
labels = np.array(["n" for i in n] + ["s" for j in s] + ["t" for k in c])
df2 = pd.DataFrame(np.array([ratios, labels]).T)
df2.columns = ['ratios', 'labels']
df2["ratios"] = pd.to_numeric(df2["ratios"], downcast="float")

sns.barplot(x='labels', y='ratios', data=df2, ax=axins, errwidth=1, capsize=0.1, ci=95,
            palette=[(0, 1, 0), (.5, .5, .5), palette["semi torus"]], saturation=0.75, alpha=0.7)

# axins.text(0, 1, r"$\it{none}$", ha="center", va="bottom", size=10)
axins.text(0, 1.2, r"$\it{cells \; only}$", ha="center", va="bottom", rotation=90, size=11)
# axins.text(1, 0.25, "spheres", ha="center", va="bottom", rotation=90, size=10)
# axins.text(2, 0.25, "semi tori", ha="center", va="bottom", rotation=90, size=10)

axins.set_ylim([0, 6])
# axins.set_yticks([0, 0.002, 0.])
# axins.set_yticklabels([])
# axins.set_xticks(range(1, 7, 1))
# axins.set_xlim([0, 4])
# axins.set_xticks([1,2,3])
axins.set_xticklabels([])
# axins.set_ylabel(r'$\frac{\mathrm{diameter}}{\mathrm{cell \;\, density}}$', fontsize=14)
axins.set_ylabel('diameter / cell density', fontsize=11)
axins.set_xlabel('')
# axins.set_xlabel('Progenitors', fontsize=11)
# axins.set_xticklabels(['none', 'o', 'c'])
axins.tick_params(axis='both', which='major', labelsize=10)


f.subplots_adjust(wspace=-0.05, hspace=-0.05)
bbox = f.get_window_extent().transformed(f.dpi_scale_trans.inverted())
plt.savefig("test.png", bbox_inches='tight', dpi=300, transparent=True)

