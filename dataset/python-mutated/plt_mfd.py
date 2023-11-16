"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.4

@author: Thomas Chartier
"""
import numpy as np
import matplotlib.pyplot as plt

def add_data(data):
    if False:
        print('Hello World!')
    x = data[0]
    y = data[1]
    plt.plot(x, y, 'k', linewidth=3, alpha=0.8, label='cumulative catalogue MFD')

def plot(x, y, lim, axis, data, path, title):
    if False:
        return 10
    '\n    x : list, bining in magnitude\n    y : list, mfd values (same length as x)\n    lim : list, 2D of len 2, [[xmin,xmax],[ymin,ymax]]\n    axis : list, 2 strings for the axis title x and y\n    data : bool or list, False or mfd value of the catalog\n    path : bool, destination path of the figure\n    title : str, title of the figure\n\n    '
    y = list(y)
    y_cum = list(np.cumsum(np.array(y[::-1])))
    y_cum = y_cum[::-1]
    plt.scatter(x, y, c='darkcyan', s=50, marker='s', alpha=0.7, label='SHERIFS incremental MFD')
    plt.plot(x, y_cum, 'darkgreen', linewidth=2, alpha=0.8, label='SHERIFS cumulative MFD')
    if not data == False:
        add_data(data)
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=180, transparent=False)
    plt.close()

def plot_bg_ft(x, ys, lim, axis, path, title):
    if False:
        i = 10
        return i + 15
    '\n    x : list, bining in magnitude\n    y : list, mfd values for whole model, faults, bg (3 x same length as x)\n    lim : list, 2D of len 2, [[xmin,xmax],[ymin,ymax]]\n    axis : list, 2 strings for the axis title x and y\n    data : bool or list, False or mfd value of the catalog\n    path : bool, destination path of the figure\n    title : str, title of the figure\n\n    '
    y = list(ys[0])
    y_cum = list(np.cumsum(np.array(y[::-1])))
    y_cum = y_cum[::-1]
    plt.plot(x, y_cum, 'darkgreen', linewidth=2, alpha=0.8, label='SHERIFS cumulative MFD')
    y = list(ys[1])
    y_cum = list(np.cumsum(np.array(y[::-1])))
    y_cum = y_cum[::-1]
    plt.plot(x, y_cum, 'red', linewidth=2, alpha=0.8, label='Faults cumulative MFD')
    y = list(ys[2])
    y_cum = list(np.cumsum(np.array(y[::-1])))
    y_cum = y_cum[::-1]
    plt.plot(x, y_cum, 'blue', linewidth=2, alpha=0.8, label='Background cumulative MFD')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=180, transparent=False)
    plt.close()

def local(x, ys, data, lim, axis, path, title):
    if False:
        return 10
    '\n    x : list, bining in magnitude\n    y : list, mfd values for whole model, faults, bg\n                and smooth before scalling (4 x same length as x)\n    lim : list, 2D of len 2, [[xmin,xmax],[ymin,ymax]]\n    axis : list, 2 strings for the axis title x and y\n    data : bool or list, False or mfd value of the catalog\n    path : bool, destination path of the figure\n    title : str, title of the figure\n\n    '
    y = list(ys[0])
    y_cum = list(np.cumsum(np.array(y[::-1])))
    y_cum = y_cum[::-1]
    plt.plot(x, y_cum, 'darkgreen', linewidth=2, alpha=0.8, label='SHERIFS cumulative MFD')
    y = list(ys[1])
    y_cum = list(np.cumsum(np.array(y[::-1])))
    y_cum = y_cum[::-1]
    plt.plot(x, y_cum, 'red', linewidth=2, alpha=0.8, label='Faults cumulative MFD')
    y = list(ys[2])
    y_cum = list(np.cumsum(np.array(y[::-1])))
    y_cum = y_cum[::-1]
    plt.plot(x, y_cum, 'blue', linewidth=2, alpha=0.8, label='Background cumulative MFD')
    if not data == False:
        add_data(data)
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=180, transparent=False)
    plt.close()