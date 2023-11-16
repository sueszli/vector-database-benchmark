"""
Overlapping densities ('ridge plot')
====================================


"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='white', rc={'axes.facecolor': (0, 0, 0, 0)})
rs = np.random.RandomState(1979)
x = rs.randn(500)
g = np.tile(list('ABCDEFGHIJ'), 50)
df = pd.DataFrame(dict(x=x, g=g))
m = df.g.map(ord)
df['x'] += m
pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
g = sns.FacetGrid(df, row='g', hue='g', aspect=15, height=0.5, palette=pal)
g.map(sns.kdeplot, 'x', bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, 'x', clip_on=False, color='w', lw=2, bw_adjust=0.5)
g.refline(y=0, linewidth=2, linestyle='-', color=None, clip_on=False)

def label(x, color, label):
    if False:
        for i in range(10):
            print('nop')
    ax = plt.gca()
    ax.text(0, 0.2, label, fontweight='bold', color=color, ha='left', va='center', transform=ax.transAxes)
g.map(label, 'x')
g.figure.subplots_adjust(hspace=-0.25)
g.set_titles('')
g.set(yticks=[], ylabel='')
g.despine(bottom=True, left=True)