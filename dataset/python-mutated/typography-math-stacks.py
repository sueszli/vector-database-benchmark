import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

def plot(family):
    if False:
        i = 10
        return i + 15
    plt.rcParams['mathtext.fontset'] = family
    fig = plt.figure(figsize=(3, 1.75), dpi=100)
    ax = plt.subplot(1, 1, 1, frameon=False, xlim=(-1, 1), ylim=(-0.5, 1.5), xticks=[], yticks=[])
    ax.text(0, 0.0, '$\\frac{\\pi}{4} = \\sum_{k=0}^\\infty\\frac{(-1)^k}{2k+1}$', size=32, ha='center', va='bottom')
    ax.text(0, -0.1, 'mathtext.fontset = "%s"' % family, size=14, ha='center', va='top', family='Roboto Condensed', color='0.5')
    plt.savefig('../../figures/typography/typography-math-%s.pdf' % family, dpi=600)
    plt.show()
plot('cm')
plot('stix')
plot('stixsans')
plot('dejavusans')
plot('dejavuserif')
plot('custom')