import matplotlib
import numpy as np
matplotlib.use('module://matplotlib_pyodide.html5_canvas_backend')
from matplotlib import pyplot as plt

def canvas_custom_font():
    if False:
        while True:
            i = 10
    f = {'fontname': 'cmsy10'}
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    plt.figure()
    plt.title('A simple Sine Curve', **f)
    plt.plot(t, s, linewidth=1.0, marker=11)
    plt.plot(t, t)
    plt.grid(True)
    plt.show()
    plt.close('all')
    plt.clf()