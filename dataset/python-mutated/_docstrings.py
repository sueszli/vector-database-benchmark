import re
import pydoc
from .external.docscrape import NumpyDocString

class DocstringComponents:
    regexp = re.compile('\\n((\\n|.)+)\\n\\s*', re.MULTILINE)

    def __init__(self, comp_dict, strip_whitespace=True):
        if False:
            while True:
                i = 10
        'Read entries from a dict, optionally stripping outer whitespace.'
        if strip_whitespace:
            entries = {}
            for (key, val) in comp_dict.items():
                m = re.match(self.regexp, val)
                if m is None:
                    entries[key] = val
                else:
                    entries[key] = m.group(1)
        else:
            entries = comp_dict.copy()
        self.entries = entries

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        'Provide dot access to entries for clean raw docstrings.'
        if attr in self.entries:
            return self.entries[attr]
        else:
            try:
                return self.__getattribute__(attr)
            except AttributeError as err:
                if __debug__:
                    raise err
                else:
                    pass

    @classmethod
    def from_nested_components(cls, **kwargs):
        if False:
            print('Hello World!')
        'Add multiple sub-sets of components.'
        return cls(kwargs, strip_whitespace=False)

    @classmethod
    def from_function_params(cls, func):
        if False:
            for i in range(10):
                print('nop')
        'Use the numpydoc parser to extract components from existing func.'
        params = NumpyDocString(pydoc.getdoc(func))['Parameters']
        comp_dict = {}
        for p in params:
            name = p.name
            type = p.type
            desc = '\n    '.join(p.desc)
            comp_dict[name] = f'{name} : {type}\n    {desc}'
        return cls(comp_dict)
_core_params = dict(data='\ndata : :class:`pandas.DataFrame`, :class:`numpy.ndarray`, mapping, or sequence\n    Input data structure. Either a long-form collection of vectors that can be\n    assigned to named variables or a wide-form dataset that will be internally\n    reshaped.\n    ', xy='\nx, y : vectors or keys in ``data``\n    Variables that specify positions on the x and y axes.\n    ', hue='\nhue : vector or key in ``data``\n    Semantic variable that is mapped to determine the color of plot elements.\n    ', palette='\npalette : string, list, dict, or :class:`matplotlib.colors.Colormap`\n    Method for choosing the colors to use when mapping the ``hue`` semantic.\n    String values are passed to :func:`color_palette`. List or dict values\n    imply categorical mapping, while a colormap object implies numeric mapping.\n    ', hue_order='\nhue_order : vector of strings\n    Specify the order of processing and plotting for categorical levels of the\n    ``hue`` semantic.\n    ', hue_norm='\nhue_norm : tuple or :class:`matplotlib.colors.Normalize`\n    Either a pair of values that set the normalization range in data units\n    or an object that will map from data units into a [0, 1] interval. Usage\n    implies numeric mapping.\n    ', color='\ncolor : :mod:`matplotlib color <matplotlib.colors>`\n    Single color specification for when hue mapping is not used. Otherwise, the\n    plot will try to hook into the matplotlib property cycle.\n    ', ax='\nax : :class:`matplotlib.axes.Axes`\n    Pre-existing axes for the plot. Otherwise, call :func:`matplotlib.pyplot.gca`\n    internally.\n    ')
_core_returns = dict(ax='\n:class:`matplotlib.axes.Axes`\n    The matplotlib axes containing the plot.\n    ', facetgrid='\n:class:`FacetGrid`\n    An object managing one or more subplots that correspond to conditional data\n    subsets with convenient methods for batch-setting of axes attributes.\n    ', jointgrid='\n:class:`JointGrid`\n    An object managing multiple subplots that correspond to joint and marginal axes\n    for plotting a bivariate relationship or distribution.\n    ', pairgrid='\n:class:`PairGrid`\n    An object managing multiple subplots that correspond to joint and marginal axes\n    for pairwise combinations of multiple variables in a dataset.\n    ')
_seealso_blurbs = dict(scatterplot='\nscatterplot : Plot data using points.\n    ', lineplot='\nlineplot : Plot data using lines.\n    ', displot='\ndisplot : Figure-level interface to distribution plot functions.\n    ', histplot='\nhistplot : Plot a histogram of binned counts with optional normalization or smoothing.\n    ', kdeplot='\nkdeplot : Plot univariate or bivariate distributions using kernel density estimation.\n    ', ecdfplot='\necdfplot : Plot empirical cumulative distribution functions.\n    ', rugplot='\nrugplot : Plot a tick at each observation value along the x and/or y axes.\n    ', stripplot='\nstripplot : Plot a categorical scatter with jitter.\n    ', swarmplot='\nswarmplot : Plot a categorical scatter with non-overlapping points.\n    ', violinplot='\nviolinplot : Draw an enhanced boxplot using kernel density estimation.\n    ', pointplot='\npointplot : Plot point estimates and CIs using markers and lines.\n    ', jointplot='\njointplot : Draw a bivariate plot with univariate marginal distributions.\n    ', pairplot='\njointplot : Draw multiple bivariate plots with univariate marginal distributions.\n    ', jointgrid='\nJointGrid : Set up a figure with joint and marginal views on bivariate data.\n    ', pairgrid='\nPairGrid : Set up a figure with joint and marginal views on multiple variables.\n    ')
_core_docs = dict(params=DocstringComponents(_core_params), returns=DocstringComponents(_core_returns), seealso=DocstringComponents(_seealso_blurbs))