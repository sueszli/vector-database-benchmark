import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rc('font', family='Roboto')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rc('axes', labelsize='medium', titlesize='medium')
fig = plt.figure(figsize=(6, 2))
ax = plt.subplot(3, 1, 1, xlim=[0.0, 1.0], ylim=[0, 1])
ax.patch.set_facecolor('none')
ax.text(0.5, 0.1, 'linear scale', transform=ax.transAxes, horizontalalignment='center', verticalalignment='bottom')
ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.tick_params(axis='both', which='major')

def forward(x):
    if False:
        return 10
    return x ** 2

def inverse(x):
    if False:
        return 10
    return x ** (1 / 2)
ax = plt.subplot(3, 1, 2, xlim=[0.1, 1.0], ylim=[0, 1])
ax.patch.set_facecolor('none')
ax.text(0.5, 0.1, '$x^2$ scale', transform=ax.transAxes, horizontalalignment='center', verticalalignment='bottom')
ax.set_xscale('function', functions=(forward, inverse))
ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax = plt.subplot(3, 1, 3, xlim=[0.1, 1.0], ylim=[0, 1])
ax.patch.set_facecolor('none')
ax.text(0.5, 0.1, '$\\sqrt{x}$ scale', transform=ax.transAxes, horizontalalignment='center', verticalalignment='bottom')
ax.set_xscale('function', functions=(inverse, forward))
ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
plt.tight_layout()
plt.savefig('../../figures/scales-projections/scales-custom.pdf')
plt.show()