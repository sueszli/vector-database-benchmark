from collections import OrderedDict
from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs
np = optional_imports.get_module('numpy')
scp = optional_imports.get_module('scipy')
sch = optional_imports.get_module('scipy.cluster.hierarchy')
scs = optional_imports.get_module('scipy.spatial')

def create_dendrogram(X, orientation='bottom', labels=None, colorscale=None, distfun=None, linkagefun=lambda x: sch.linkage(x, 'complete'), hovertext=None, color_threshold=None):
    if False:
        i = 10
        return i + 15
    "\n    Function that returns a dendrogram Plotly figure object. This is a thin\n    wrapper around scipy.cluster.hierarchy.dendrogram.\n\n    See also https://dash.plot.ly/dash-bio/clustergram.\n\n    :param (ndarray) X: Matrix of observations as array of arrays\n    :param (str) orientation: 'top', 'right', 'bottom', or 'left'\n    :param (list) labels: List of axis category labels(observation labels)\n    :param (list) colorscale: Optional colorscale for the dendrogram tree.\n                              Requires 8 colors to be specified, the 7th of\n                              which is ignored.  With scipy>=1.5.0, the 2nd, 3rd\n                              and 6th are used twice as often as the others.\n                              Given a shorter list, the missing values are\n                              replaced with defaults and with a longer list the\n                              extra values are ignored.\n    :param (function) distfun: Function to compute the pairwise distance from\n                               the observations\n    :param (function) linkagefun: Function to compute the linkage matrix from\n                               the pairwise distances\n    :param (list[list]) hovertext: List of hovertext for constituent traces of dendrogram\n                               clusters\n    :param (double) color_threshold: Value at which the separation of clusters will be made\n\n    Example 1: Simple bottom oriented dendrogram\n\n    >>> from plotly.figure_factory import create_dendrogram\n\n    >>> import numpy as np\n\n    >>> X = np.random.rand(10,10)\n    >>> fig = create_dendrogram(X)\n    >>> fig.show()\n\n    Example 2: Dendrogram to put on the left of the heatmap\n\n    >>> from plotly.figure_factory import create_dendrogram\n\n    >>> import numpy as np\n\n    >>> X = np.random.rand(5,5)\n    >>> names = ['Jack', 'Oxana', 'John', 'Chelsea', 'Mark']\n    >>> dendro = create_dendrogram(X, orientation='right', labels=names)\n    >>> dendro.update_layout({'width':700, 'height':500}) # doctest: +SKIP\n    >>> dendro.show()\n\n    Example 3: Dendrogram with Pandas\n\n    >>> from plotly.figure_factory import create_dendrogram\n\n    >>> import numpy as np\n    >>> import pandas as pd\n\n    >>> Index= ['A','B','C','D','E','F','G','H','I','J']\n    >>> df = pd.DataFrame(abs(np.random.randn(10, 10)), index=Index)\n    >>> fig = create_dendrogram(df, labels=Index)\n    >>> fig.show()\n    "
    if not scp or not scs or (not sch):
        raise ImportError('FigureFactory.create_dendrogram requires scipy,                             scipy.spatial and scipy.hierarchy')
    s = X.shape
    if len(s) != 2:
        exceptions.PlotlyError('X should be 2-dimensional array.')
    if distfun is None:
        distfun = scs.distance.pdist
    dendrogram = _Dendrogram(X, orientation, labels, colorscale, distfun=distfun, linkagefun=linkagefun, hovertext=hovertext, color_threshold=color_threshold)
    return graph_objs.Figure(data=dendrogram.data, layout=dendrogram.layout)

class _Dendrogram(object):
    """Refer to FigureFactory.create_dendrogram() for docstring."""

    def __init__(self, X, orientation='bottom', labels=None, colorscale=None, width=np.inf, height=np.inf, xaxis='xaxis', yaxis='yaxis', distfun=None, linkagefun=lambda x: sch.linkage(x, 'complete'), hovertext=None, color_threshold=None):
        if False:
            for i in range(10):
                print('nop')
        self.orientation = orientation
        self.labels = labels
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.data = []
        self.leaves = []
        self.sign = {self.xaxis: 1, self.yaxis: 1}
        self.layout = {self.xaxis: {}, self.yaxis: {}}
        if self.orientation in ['left', 'bottom']:
            self.sign[self.xaxis] = 1
        else:
            self.sign[self.xaxis] = -1
        if self.orientation in ['right', 'bottom']:
            self.sign[self.yaxis] = 1
        else:
            self.sign[self.yaxis] = -1
        if distfun is None:
            distfun = scs.distance.pdist
        (dd_traces, xvals, yvals, ordered_labels, leaves) = self.get_dendrogram_traces(X, colorscale, distfun, linkagefun, hovertext, color_threshold)
        self.labels = ordered_labels
        self.leaves = leaves
        yvals_flat = yvals.flatten()
        xvals_flat = xvals.flatten()
        self.zero_vals = []
        for i in range(len(yvals_flat)):
            if yvals_flat[i] == 0.0 and xvals_flat[i] not in self.zero_vals:
                self.zero_vals.append(xvals_flat[i])
        if len(self.zero_vals) > len(yvals) + 1:
            l_border = int(min(self.zero_vals))
            r_border = int(max(self.zero_vals))
            correct_leaves_pos = range(l_border, r_border + 1, int((r_border - l_border) / len(yvals)))
            self.zero_vals = [v for v in correct_leaves_pos]
        self.zero_vals.sort()
        self.layout = self.set_figure_layout(width, height)
        self.data = dd_traces

    def get_color_dict(self, colorscale):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns colorscale used for dendrogram tree clusters.\n\n        :param (list) colorscale: Colors to use for the plot in rgb format.\n        :rtype (dict): A dict of default colors mapped to the user colorscale.\n\n        '
        d = {'r': 'red', 'g': 'green', 'b': 'blue', 'c': 'cyan', 'm': 'magenta', 'y': 'yellow', 'k': 'black', 'w': 'white'}
        default_colors = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        if colorscale is None:
            rgb_colorscale = ['rgb(0,116,217)', 'rgb(35,205,205)', 'rgb(61,153,112)', 'rgb(40,35,35)', 'rgb(133,20,75)', 'rgb(255,65,54)', 'rgb(255,255,255)', 'rgb(255,220,0)']
        else:
            rgb_colorscale = colorscale
        for i in range(len(default_colors.keys())):
            k = list(default_colors.keys())[i]
            if i < len(rgb_colorscale):
                default_colors[k] = rgb_colorscale[i]
        new_old_color_map = [('C0', 'b'), ('C1', 'g'), ('C2', 'r'), ('C3', 'c'), ('C4', 'm'), ('C5', 'y'), ('C6', 'k'), ('C7', 'g'), ('C8', 'r'), ('C9', 'c')]
        for (nc, oc) in new_old_color_map:
            try:
                default_colors[nc] = default_colors[oc]
            except KeyError:
                default_colors[n] = 'rgb(0,116,217)'
        return default_colors

    def set_axis_layout(self, axis_key):
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets and returns default axis object for dendrogram figure.\n\n        :param (str) axis_key: E.g., 'xaxis', 'xaxis1', 'yaxis', yaxis1', etc.\n        :rtype (dict): An axis_key dictionary with set parameters.\n\n        "
        axis_defaults = {'type': 'linear', 'ticks': 'outside', 'mirror': 'allticks', 'rangemode': 'tozero', 'showticklabels': True, 'zeroline': False, 'showgrid': False, 'showline': True}
        if len(self.labels) != 0:
            axis_key_labels = self.xaxis
            if self.orientation in ['left', 'right']:
                axis_key_labels = self.yaxis
            if axis_key_labels not in self.layout:
                self.layout[axis_key_labels] = {}
            self.layout[axis_key_labels]['tickvals'] = [zv * self.sign[axis_key] for zv in self.zero_vals]
            self.layout[axis_key_labels]['ticktext'] = self.labels
            self.layout[axis_key_labels]['tickmode'] = 'array'
        self.layout[axis_key].update(axis_defaults)
        return self.layout[axis_key]

    def set_figure_layout(self, width, height):
        if False:
            i = 10
            return i + 15
        '\n        Sets and returns default layout object for dendrogram figure.\n\n        '
        self.layout.update({'showlegend': False, 'autosize': False, 'hovermode': 'closest', 'width': width, 'height': height})
        self.set_axis_layout(self.xaxis)
        self.set_axis_layout(self.yaxis)
        return self.layout

    def get_dendrogram_traces(self, X, colorscale, distfun, linkagefun, hovertext, color_threshold):
        if False:
            print('Hello World!')
        "\n        Calculates all the elements needed for plotting a dendrogram.\n\n        :param (ndarray) X: Matrix of observations as array of arrays\n        :param (list) colorscale: Color scale for dendrogram tree clusters\n        :param (function) distfun: Function to compute the pairwise distance\n                                   from the observations\n        :param (function) linkagefun: Function to compute the linkage matrix\n                                      from the pairwise distances\n        :param (list) hovertext: List of hovertext for constituent traces of dendrogram\n        :rtype (tuple): Contains all the traces in the following order:\n            (a) trace_list: List of Plotly trace objects for dendrogram tree\n            (b) icoord: All X points of the dendrogram tree as array of arrays\n                with length 4\n            (c) dcoord: All Y points of the dendrogram tree as array of arrays\n                with length 4\n            (d) ordered_labels: leaf labels in the order they are going to\n                appear on the plot\n            (e) P['leaves']: left-to-right traversal of the leaves\n\n        "
        d = distfun(X)
        Z = linkagefun(d)
        P = sch.dendrogram(Z, orientation=self.orientation, labels=self.labels, no_plot=True, color_threshold=color_threshold)
        icoord = scp.array(P['icoord'])
        dcoord = scp.array(P['dcoord'])
        ordered_labels = scp.array(P['ivl'])
        color_list = scp.array(P['color_list'])
        colors = self.get_color_dict(colorscale)
        trace_list = []
        for i in range(len(icoord)):
            if self.orientation in ['top', 'bottom']:
                xs = icoord[i]
            else:
                xs = dcoord[i]
            if self.orientation in ['top', 'bottom']:
                ys = dcoord[i]
            else:
                ys = icoord[i]
            color_key = color_list[i]
            hovertext_label = None
            if hovertext:
                hovertext_label = hovertext[i]
            trace = dict(type='scatter', x=np.multiply(self.sign[self.xaxis], xs), y=np.multiply(self.sign[self.yaxis], ys), mode='lines', marker=dict(color=colors[color_key]), text=hovertext_label, hoverinfo='text')
            try:
                x_index = int(self.xaxis[-1])
            except ValueError:
                x_index = ''
            try:
                y_index = int(self.yaxis[-1])
            except ValueError:
                y_index = ''
            trace['xaxis'] = f'x{x_index}'
            trace['yaxis'] = f'y{y_index}'
            trace_list.append(trace)
        return (trace_list, icoord, dcoord, ordered_labels, P['leaves'])