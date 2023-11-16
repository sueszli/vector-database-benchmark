"""
======================
Demo CurveLinear Grid2
======================

Custom grid and ticklines.

This example demonstrates how to use GridHelperCurveLinear to define
custom grids and ticklines by applying a transformation on the grid.
As showcase on the plot, a 5x5 matrix is displayed on the axes.
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple, MaxNLocator
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear

def curvelinear_test1(fig):
    if False:
        print('Hello World!')
    'Grid for custom transform.'

    def tr(x, y):
        if False:
            for i in range(10):
                print('nop')
        return (np.sign(x) * abs(x) ** 0.5, y)

    def inv_tr(x, y):
        if False:
            i = 10
            return i + 15
        return (np.sign(x) * x ** 2, y)
    grid_helper = GridHelperCurveLinear((tr, inv_tr), extreme_finder=ExtremeFinderSimple(20, 20), grid_locator1=MaxNLocator(nbins=6), grid_locator2=MaxNLocator(nbins=6))
    ax1 = fig.add_subplot(axes_class=Axes, grid_helper=grid_helper)
    ax1.imshow(np.arange(25).reshape(5, 5), vmax=50, cmap=plt.cm.gray_r, origin='lower')
if __name__ == '__main__':
    fig = plt.figure(figsize=(7, 4))
    curvelinear_test1(fig)
    plt.show()