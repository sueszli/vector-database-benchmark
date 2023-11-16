"""
Implements histogram with vertical lines to help with balanced binning.
"""
import numpy as np
from yellowbrick.target.base import TargetVisualizer
from yellowbrick.exceptions import YellowbrickValueError

class BalancedBinningReference(TargetVisualizer):
    """
    BalancedBinningReference generates a histogram with vertical lines
    showing the recommended value point to bin your data so they can be evenly
    distributed in each bin.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        This is inherited from FeatureVisualizer and is defined within
        ``BalancedBinningReference``.

    target : string, default: "y"
        The name of the ``y`` variable

    bins : number of bins to generate the histogram, default: 4

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    bin_edges_ : binning reference values

    Examples
    --------
    >>> visualizer = BalancedBinningReference()
    >>> visualizer.fit(y)
    >>> visualizer.show()


    Notes
    -----
    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    def __init__(self, ax=None, target=None, bins=4, **kwargs):
        if False:
            i = 10
            return i + 15
        super(BalancedBinningReference, self).__init__(ax, **kwargs)
        self.target = target
        self.bins = bins

    def draw(self, y, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Draws a histogram with the reference value for binning as vertical\n        lines.\n\n        Parameters\n        ----------\n        y : an array of one dimension or a pandas Series\n        '
        (hist, bin_edges) = np.histogram(y, bins=self.bins)
        self.bin_edges_ = bin_edges
        self.ax.hist(y, bins=self.bins, color=kwargs.pop('color', '#6897bb'), **kwargs)
        self.ax.vlines(bin_edges, 0, max(hist), colors=kwargs.pop('colors', 'r'))
        return self.ax

    def fit(self, y, **kwargs):
        if False:
            return 10
        '\n        Sets up y for the histogram and checks to\n        ensure that ``y`` is of the correct data type.\n        Fit calls draw.\n\n        Parameters\n        ----------\n        y : an array of one dimension or a pandas Series\n\n        kwargs : dict\n            keyword arguments passed to scikit-learn API.\n\n        '
        if y.ndim > 1:
            raise YellowbrickValueError('y needs to be an array or Series with one dimension')
        if self.target is None:
            self.target = 'y'
        self.draw(y)
        return self

    def finalize(self, **kwargs):
        if False:
            return 10
        "\n        Adds the x-axis label and manages the tick labels to ensure they're visible.\n\n        Parameters\n        ----------\n        kwargs: generic keyword arguments.\n\n        Notes\n        -----\n        Generally this method is called from show and not directly by the user.\n        "
        self.ax.set_xlabel(self.target)
        for tk in self.ax.get_xticklabels():
            tk.set_visible(True)
        for tk in self.ax.get_yticklabels():
            tk.set_visible(True)

def balanced_binning_reference(y, ax=None, target='y', bins=4, show=True, **kwargs):
    if False:
        while True:
            i = 10
    '\n    BalancedBinningReference generates a histogram with vertical lines\n    showing the recommended value point to bin your data so they can be evenly\n    distributed in each bin.\n\n    Parameters\n    ----------\n    y : an array of one dimension or a pandas Series\n\n    ax : matplotlib Axes, default: None\n        This is inherited from FeatureVisualizer and is defined within\n        ``BalancedBinningReference``.\n\n    target : string, default: "y"\n        The name of the ``y`` variable\n\n    bins : number of bins to generate the histogram, default: 4\n\n    show : bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()``. However, you\n        cannot call ``plt.savefig`` from this signature, nor ``clear_figure``. If False,\n        simply calls ``finalize()``.\n\n    kwargs : dict\n        Keyword arguments that are passed to the base class and may influence\n        the visualization as defined in other Visualizers.\n\n    Returns\n    -------\n    visualizer : BalancedBinningReference\n        Returns fitted visualizer\n    '
    visualizer = BalancedBinningReference(ax=ax, bins=bins, target=target, **kwargs)
    visualizer.fit(y)
    if show:
        visualizer.show()
    else:
        visualizer.finalize()
    return visualizer