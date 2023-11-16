"""
.. redirect-from:: /users/customizing
.. redirect-from:: /tutorials/introductory/customizing

.. _customizing:

=====================================================
Customizing Matplotlib with style sheets and rcParams
=====================================================

Tips for customizing the properties and default styles of Matplotlib.

There are three ways to customize Matplotlib:

1. :ref:`Setting rcParams at runtime<customizing-with-dynamic-rc-settings>`.
2. :ref:`Using style sheets<customizing-with-style-sheets>`.
3. :ref:`Changing your matplotlibrc file<customizing-with-matplotlibrc-files>`.

Setting rcParams at runtime takes precedence over style sheets, style
sheets take precedence over :file:`matplotlibrc` files.

.. _customizing-with-dynamic-rc-settings:

Runtime rc settings
===================

You can dynamically change the default rc (runtime configuration)
settings in a python script or interactively from the python shell. All
rc settings are stored in a dictionary-like variable called
:data:`matplotlib.rcParams`, which is global to the matplotlib package.
See `matplotlib.rcParams` for a full list of configurable rcParams.
rcParams can be modified directly, for example:
"""
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linestyle'] = '--'
data = np.random.randn(50)
plt.plot(data)
mpl.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])
plt.plot(data)
mpl.rc('lines', linewidth=4, linestyle='-.')
plt.plot(data)
with mpl.rc_context({'lines.linewidth': 2, 'lines.linestyle': ':'}):
    plt.plot(data)

@mpl.rc_context({'lines.linewidth': 3, 'lines.linestyle': '-'})
def plotting_function():
    if False:
        while True:
            i = 10
    plt.plot(data)
plotting_function()
plt.style.use('ggplot')
print(plt.style.available)
with plt.style.context('dark_background'):
    plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')
plt.show()