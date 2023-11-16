"""
Implementations of frequency distributions for text visualization
"""
import numpy as np
from operator import itemgetter
from yellowbrick.text.base import TextVisualizer
from yellowbrick.exceptions import YellowbrickValueError

def freqdist(features, X, y=None, ax=None, n=50, orient='h', color=None, show=True, **kwargs):
    if False:
        while True:
            i = 10
    "Displays frequency distribution plot for text.\n\n    This helper function is a quick wrapper to utilize the FreqDist\n    Visualizer (Transformer) for one-off analysis.\n\n    Parameters\n    ----------\n\n    features : list, default: None\n        The list of feature names from the vectorizer, ordered by index. E.g.\n        a lexicon that specifies the unique vocabulary of the corpus. This\n        can be typically fetched using the ``get_feature_names()`` method of\n        the transformer in Scikit-Learn.\n\n    X: ndarray or DataFrame of shape n x m\n        A matrix of n instances with m features. In the case of text,\n        X is a list of list of already preprocessed words\n\n    y: ndarray or Series of length n\n        An array or series of target or class values\n\n    ax : matplotlib axes, default: None\n        The axes to plot the figure on.\n\n    n: integer, default: 50\n        Top N tokens to be plotted.\n\n    orient : 'h' or 'v', default: 'h'\n        Specifies a horizontal or vertical bar chart.\n\n    color : string\n        Specify color for bars\n\n    show: bool, default: True\n        If True, calls ``show()``, which in turn calls ``plt.show()`` however\n        you cannot call ``plt.savefig`` from this signature, nor\n        ``clear_figure``. If False, simply calls ``finalize()``\n\n    kwargs: dict\n        Keyword arguments passed to the super class.\n\n    Returns\n    -------\n    visualizer: FreqDistVisualizer\n        Returns the fitted, finalized visualizer\n    "
    viz = FreqDistVisualizer(features, ax=ax, n=n, orient=orient, color=color, **kwargs)
    viz.fit(X, y, **kwargs)
    viz.transform(X)
    if show:
        viz.show()
    else:
        viz.finalize()
    return viz

class FrequencyVisualizer(TextVisualizer):
    """
    A frequency distribution tells us the frequency of each vocabulary
    item in the text. In general, it could count any kind of observable
    event. It is a distribution because it tells us how the total
    number of word tokens in the text are distributed across the
    vocabulary items.


    Parameters
    ----------
    features : list, default: None
        The list of feature names from the vectorizer, ordered by index. E.g.
        a lexicon that specifies the unique vocabulary of the corpus. This
        can be typically fetched using the ``get_feature_names()`` method of
        the transformer in Scikit-Learn.

    ax : matplotlib axes, default: None
        The axes to plot the figure on.

    n: integer, default: 50
        Top N tokens to be plotted.

    orient : 'h' or 'v', default: 'h'
        Specifies a horizontal or vertical bar chart.

    color : string
        Specify color for bars

    kwargs : dict
        Pass any additional keyword arguments to the super class.

    These parameters can be influenced later on in the visualization
    process, but can and should be set as early as possible.
    """

    def __init__(self, features, ax=None, n=50, orient='h', color=None, **kwargs):
        if False:
            while True:
                i = 10
        super(FreqDistVisualizer, self).__init__(ax=ax, **kwargs)
        orient = orient.lower().strip()
        if orient not in {'h', 'v'}:
            raise YellowbrickValueError("Orientation must be 'h' or 'v'")
        self.N = n
        self.features = features
        self.color = color
        self.orient = orient

    def count(self, X):
        if False:
            return 10
        '\n        Called from the fit method, this method gets all the\n        words from the corpus and their corresponding frequency\n        counts.\n\n        Parameters\n        ----------\n\n        X : ndarray or masked ndarray\n            Pass in the matrix of vectorized documents, can be masked in\n            order to sum the word frequencies for only a subset of documents.\n\n        Returns\n        -------\n\n        counts : array\n            A vector containing the counts of all words in X (columns)\n\n        '
        return np.squeeze(np.asarray(X.sum(axis=0)))

    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        The fit method is the primary drawing input for the frequency\n        distribution visualization. It requires vectorized lists of\n        documents and a list of features, which are the actual words\n        from the original corpus (needed to label the x-axis ticks).\n\n        Parameters\n        ----------\n        X : ndarray or DataFrame of shape n x m\n            A matrix of n instances with m features representing the corpus\n            of frequency vectorized documents.\n\n        y : ndarray or DataFrame of shape n\n            Labels for the documents for conditional frequency distribution.\n\n        Notes\n        -----\n        .. note:: Text documents must be vectorized before ``fit()``.\n        '
        if y is not None:
            self.conditional_freqdist_ = {}
            self.classes_ = [str(label) for label in set(y)]
            for label in self.classes_:
                self.conditional_freqdist_[label] = self.count(X[y == label])
        else:
            self.conditional_freqdist_ = None
        self.freqdist_ = self.count(X)
        self.sorted_ = self.freqdist_.argsort()[::-1]
        self.vocab_ = self.freqdist_.shape[0]
        self.words_ = self.freqdist_.sum()
        self.hapaxes_ = sum((1 for c in self.freqdist_ if c == 1))
        self.draw()
        return self

    def draw(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Called from the fit method, this method creates the canvas and\n        draws the distribution plot on it.\n\n        Parameters\n        ----------\n        kwargs: generic keyword arguments.\n\n        '
        bins = np.arange(self.N)
        words = [self.features[i] for i in self.sorted_[:self.N]]
        freqs = {}
        if self.conditional_freqdist_:
            for (label, values) in sorted(self.conditional_freqdist_.items(), key=itemgetter(0)):
                freqs[label] = [values[i] for i in self.sorted_[:self.N]]
        else:
            freqs['corpus'] = [self.freqdist_[i] for i in self.sorted_[:self.N]]
        if self.orient == 'h':
            for (label, freq) in freqs.items():
                self.ax.barh(bins, freq, label=label, color=self.color, align='center')
            self.ax.set_yticks(bins)
            self.ax.set_yticklabels(words)
            self.ax.invert_yaxis()
            self.ax.yaxis.grid(False)
            self.ax.xaxis.grid(True)
        elif self.orient == 'v':
            for (label, freq) in freqs.items():
                self.ax.bar(bins, freq, label=label, color=self.color, align='edge')
            self.ax.set_xticks(bins)
            self.ax.set_xticklabels(words, rotation=90)
            self.ax.yaxis.grid(True)
            self.ax.xaxis.grid(False)
        else:
            raise YellowbrickValueError("Orientation must be 'h' or 'v'")
        return self.ax

    def finalize(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        The finalize method executes any subclass-specific axes\n        finalization steps. The user calls show & show calls finalize.\n\n        Parameters\n        ----------\n        kwargs: generic keyword arguments.\n\n        '
        self.set_title('Frequency Distribution of Top {} tokens'.format(self.N))
        infolabel = 'vocab: {:,}\nwords: {:,}\nhapax: {:,}'.format(self.vocab_, self.words_, self.hapaxes_)
        self.ax.text(0.68, 0.97, infolabel, transform=self.ax.transAxes, fontsize=9, verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
        self.ax.legend(loc='upper right', frameon=True)
FreqDistVisualizer = FrequencyVisualizer