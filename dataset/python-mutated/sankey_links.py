"""
======================================
Long chain of connections using Sankey
======================================

Demonstrate/test the Sankey class by producing a long chain of connections.
"""
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
links_per_side = 6

def side(sankey, n=1):
    if False:
        for i in range(10):
            print('nop')
    'Generate a side chain.'
    prior = len(sankey.diagrams)
    for i in range(0, 2 * n, 2):
        sankey.add(flows=[1, -1], orientations=[-1, -1], patchlabel=str(prior + i), prior=prior + i - 1, connect=(1, 0), alpha=0.5)
        sankey.add(flows=[1, -1], orientations=[1, 1], patchlabel=str(prior + i + 1), prior=prior + i, connect=(1, 0), alpha=0.5)

def corner(sankey):
    if False:
        print('Hello World!')
    'Generate a corner link.'
    prior = len(sankey.diagrams)
    sankey.add(flows=[1, -1], orientations=[0, 1], patchlabel=str(prior), facecolor='k', prior=prior - 1, connect=(1, 0), alpha=0.5)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title='Why would you want to do this?\n(But you could.)')
sankey = Sankey(ax=ax, unit=None)
sankey.add(flows=[1, -1], orientations=[0, 1], patchlabel='0', facecolor='k', rotation=45)
side(sankey, n=links_per_side)
corner(sankey)
side(sankey, n=links_per_side)
corner(sankey)
side(sankey, n=links_per_side)
corner(sankey)
side(sankey, n=links_per_side)
sankey.finish()
plt.show()