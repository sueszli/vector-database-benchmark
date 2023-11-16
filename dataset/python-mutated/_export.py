"""
This module defines export functions for decision trees.
"""
from collections.abc import Iterable
from io import StringIO
from numbers import Integral
import numpy as np
from ..base import is_classifier
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from ..utils.validation import check_array, check_is_fitted
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim

def _color_brew(n):
    if False:
        for i in range(10):
            print('nop')
    'Generate n colors with equally spaced hues.\n\n    Parameters\n    ----------\n    n : int\n        The number of colors required.\n\n    Returns\n    -------\n    color_list : list, length n\n        List of n tuples of form (R, G, B) being the components of each color.\n    '
    color_list = []
    (s, v) = (0.75, 0.9)
    c = s * v
    m = v - c
    for h in np.arange(25, 385, 360.0 / n).astype(int):
        h_bar = h / 60.0
        x = c * (1 - abs(h_bar % 2 - 1))
        rgb = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x), (c, x, 0)]
        (r, g, b) = rgb[int(h_bar)]
        rgb = [int(255 * (r + m)), int(255 * (g + m)), int(255 * (b + m))]
        color_list.append(rgb)
    return color_list

class Sentinel:

    def __repr__(self):
        if False:
            print('Hello World!')
        return '"tree.dot"'
SENTINEL = Sentinel()

@validate_params({'decision_tree': [DecisionTreeClassifier, DecisionTreeRegressor], 'max_depth': [Interval(Integral, 0, None, closed='left'), None], 'feature_names': ['array-like', None], 'class_names': ['array-like', 'boolean', None], 'label': [StrOptions({'all', 'root', 'none'})], 'filled': ['boolean'], 'impurity': ['boolean'], 'node_ids': ['boolean'], 'proportion': ['boolean'], 'rounded': ['boolean'], 'precision': [Interval(Integral, 0, None, closed='left'), None], 'ax': 'no_validation', 'fontsize': [Interval(Integral, 0, None, closed='left'), None]}, prefer_skip_nested_validation=True)
def plot_tree(decision_tree, *, max_depth=None, feature_names=None, class_names=None, label='all', filled=False, impurity=True, node_ids=False, proportion=False, rounded=False, precision=3, ax=None, fontsize=None):
    if False:
        while True:
            i = 10
    'Plot a decision tree.\n\n    The sample counts that are shown are weighted with any sample_weights that\n    might be present.\n\n    The visualization is fit automatically to the size of the axis.\n    Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control\n    the size of the rendering.\n\n    Read more in the :ref:`User Guide <tree>`.\n\n    .. versionadded:: 0.21\n\n    Parameters\n    ----------\n    decision_tree : decision tree regressor or classifier\n        The decision tree to be plotted.\n\n    max_depth : int, default=None\n        The maximum depth of the representation. If None, the tree is fully\n        generated.\n\n    feature_names : array-like of str, default=None\n        Names of each of the features.\n        If None, generic names will be used ("x[0]", "x[1]", ...).\n\n    class_names : array-like of str or True, default=None\n        Names of each of the target classes in ascending numerical order.\n        Only relevant for classification and not supported for multi-output.\n        If ``True``, shows a symbolic representation of the class name.\n\n    label : {\'all\', \'root\', \'none\'}, default=\'all\'\n        Whether to show informative labels for impurity, etc.\n        Options include \'all\' to show at every node, \'root\' to show only at\n        the top root node, or \'none\' to not show at any node.\n\n    filled : bool, default=False\n        When set to ``True``, paint nodes to indicate majority class for\n        classification, extremity of values for regression, or purity of node\n        for multi-output.\n\n    impurity : bool, default=True\n        When set to ``True``, show the impurity at each node.\n\n    node_ids : bool, default=False\n        When set to ``True``, show the ID number on each node.\n\n    proportion : bool, default=False\n        When set to ``True``, change the display of \'values\' and/or \'samples\'\n        to be proportions and percentages respectively.\n\n    rounded : bool, default=False\n        When set to ``True``, draw node boxes with rounded corners and use\n        Helvetica fonts instead of Times-Roman.\n\n    precision : int, default=3\n        Number of digits of precision for floating point in the values of\n        impurity, threshold and value attributes of each node.\n\n    ax : matplotlib axis, default=None\n        Axes to plot to. If None, use current axis. Any previous content\n        is cleared.\n\n    fontsize : int, default=None\n        Size of text font. If None, determined automatically to fit figure.\n\n    Returns\n    -------\n    annotations : list of artists\n        List containing the artists for the annotation boxes making up the\n        tree.\n\n    Examples\n    --------\n    >>> from sklearn.datasets import load_iris\n    >>> from sklearn import tree\n\n    >>> clf = tree.DecisionTreeClassifier(random_state=0)\n    >>> iris = load_iris()\n\n    >>> clf = clf.fit(iris.data, iris.target)\n    >>> tree.plot_tree(clf)\n    [...]\n    '
    check_is_fitted(decision_tree)
    exporter = _MPLTreeExporter(max_depth=max_depth, feature_names=feature_names, class_names=class_names, label=label, filled=filled, impurity=impurity, node_ids=node_ids, proportion=proportion, rounded=rounded, precision=precision, fontsize=fontsize)
    return exporter.export(decision_tree, ax=ax)

class _BaseTreeExporter:

    def __init__(self, max_depth=None, feature_names=None, class_names=None, label='all', filled=False, impurity=True, node_ids=False, proportion=False, rounded=False, precision=3, fontsize=None):
        if False:
            for i in range(10):
                print('nop')
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.class_names = class_names
        self.label = label
        self.filled = filled
        self.impurity = impurity
        self.node_ids = node_ids
        self.proportion = proportion
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize

    def get_color(self, value):
        if False:
            while True:
                i = 10
        if self.colors['bounds'] is None:
            color = list(self.colors['rgb'][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0.0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
        else:
            color = list(self.colors['rgb'][0])
            alpha = (value - self.colors['bounds'][0]) / (self.colors['bounds'][1] - self.colors['bounds'][0])
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        return '#%2x%2x%2x' % tuple(color)

    def get_fill_color(self, tree, node_id):
        if False:
            for i in range(10):
                print('nop')
        if 'rgb' not in self.colors:
            self.colors['rgb'] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                self.colors['bounds'] = (np.min(-tree.impurity), np.max(-tree.impurity))
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                self.colors['bounds'] = (np.min(tree.value), np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :] / tree.weighted_n_node_samples[node_id]
            if tree.n_classes[0] == 1:
                node_val = tree.value[node_id][0, :]
                if isinstance(node_val, Iterable) and self.colors['bounds'] is not None:
                    node_val = node_val.item()
        else:
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    def node_to_str(self, tree, node_id, criterion):
        if False:
            for i in range(10):
                print('nop')
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]
        labels = self.label == 'root' and node_id == 0 or self.label == 'all'
        characters = self.characters
        node_string = characters[-1]
        if self.node_ids:
            if labels:
                node_string += 'node '
            node_string += characters[0] + str(node_id) + characters[4]
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = 'x%s%s%s' % (characters[1], tree.feature[node_id], characters[2])
            node_string += '%s %s %s%s' % (feature, characters[3], round(tree.threshold[node_id], self.precision), characters[4])
        if self.impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion = 'friedman_mse'
            elif isinstance(criterion, _criterion.MSE) or criterion == 'squared_error':
                criterion = 'squared_error'
            elif not isinstance(criterion, str):
                criterion = 'impurity'
            if labels:
                node_string += '%s = ' % criterion
            node_string += str(round(tree.impurity[node_id], self.precision)) + characters[4]
        if labels:
            node_string += 'samples = '
        if self.proportion:
            percent = 100.0 * tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
            node_string += str(round(percent, 1)) + '%' + characters[4]
        else:
            node_string += str(tree.n_node_samples[node_id]) + characters[4]
        if self.proportion and tree.n_classes[0] != 1:
            value = value / tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += 'value = '
        if tree.n_classes[0] == 1:
            value_text = np.around(value, self.precision)
        elif self.proportion:
            value_text = np.around(value, self.precision)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            value_text = value.astype(int)
        else:
            value_text = np.around(value, self.precision)
        value_text = str(value_text.astype('S32')).replace("b'", "'")
        value_text = value_text.replace("' '", ', ').replace("'", '')
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace('[', '').replace(']', '')
        value_text = value_text.replace('\n ', characters[4])
        node_string += value_text + characters[4]
        if self.class_names is not None and tree.n_classes[0] != 1 and (tree.n_outputs == 1):
            if labels:
                node_string += 'class = '
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(value)]
            else:
                class_name = 'y%s%s%s' % (characters[1], np.argmax(value), characters[2])
            node_string += class_name
        if node_string.endswith(characters[4]):
            node_string = node_string[:-len(characters[4])]
        return node_string + characters[5]

class _DOTTreeExporter(_BaseTreeExporter):

    def __init__(self, out_file=SENTINEL, max_depth=None, feature_names=None, class_names=None, label='all', filled=False, leaves_parallel=False, impurity=True, node_ids=False, proportion=False, rotate=False, rounded=False, special_characters=False, precision=3, fontname='helvetica'):
        if False:
            return 10
        super().__init__(max_depth=max_depth, feature_names=feature_names, class_names=class_names, label=label, filled=filled, impurity=impurity, node_ids=node_ids, proportion=proportion, rounded=rounded, precision=precision)
        self.leaves_parallel = leaves_parallel
        self.out_file = out_file
        self.special_characters = special_characters
        self.fontname = fontname
        self.rotate = rotate
        if special_characters:
            self.characters = ['&#35;', '<SUB>', '</SUB>', '&le;', '<br/>', '>', '<']
        else:
            self.characters = ['#', '[', ']', '<=', '\\n', '"', '"']
        self.ranks = {'leaves': []}
        self.colors = {'bounds': None}

    def export(self, decision_tree):
        if False:
            print('Hello World!')
        if self.feature_names is not None:
            if len(self.feature_names) != decision_tree.n_features_in_:
                raise ValueError('Length of feature_names, %d does not match number of features, %d' % (len(self.feature_names), decision_tree.n_features_in_))
        self.head()
        if isinstance(decision_tree, _tree.Tree):
            self.recurse(decision_tree, 0, criterion='impurity')
        else:
            self.recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)
        self.tail()

    def tail(self):
        if False:
            i = 10
            return i + 15
        if self.leaves_parallel:
            for rank in sorted(self.ranks):
                self.out_file.write('{rank=same ; ' + '; '.join((r for r in self.ranks[rank])) + '} ;\n')
        self.out_file.write('}')

    def head(self):
        if False:
            return 10
        self.out_file.write('digraph Tree {\n')
        self.out_file.write('node [shape=box')
        rounded_filled = []
        if self.filled:
            rounded_filled.append('filled')
        if self.rounded:
            rounded_filled.append('rounded')
        if len(rounded_filled) > 0:
            self.out_file.write(', style="%s", color="black"' % ', '.join(rounded_filled))
        self.out_file.write(', fontname="%s"' % self.fontname)
        self.out_file.write('] ;\n')
        if self.leaves_parallel:
            self.out_file.write('graph [ranksep=equally, splines=polyline] ;\n')
        self.out_file.write('edge [fontname="%s"] ;\n' % self.fontname)
        if self.rotate:
            self.out_file.write('rankdir=LR ;\n')

    def recurse(self, tree, node_id, criterion, parent=None, depth=0):
        if False:
            for i in range(10):
                print('nop')
        if node_id == _tree.TREE_LEAF:
            raise ValueError('Invalid node_id %s' % _tree.TREE_LEAF)
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        if self.max_depth is None or depth <= self.max_depth:
            if left_child == _tree.TREE_LEAF:
                self.ranks['leaves'].append(str(node_id))
            elif str(depth) not in self.ranks:
                self.ranks[str(depth)] = [str(node_id)]
            else:
                self.ranks[str(depth)].append(str(node_id))
            self.out_file.write('%d [label=%s' % (node_id, self.node_to_str(tree, node_id, criterion)))
            if self.filled:
                self.out_file.write(', fillcolor="%s"' % self.get_fill_color(tree, node_id))
            self.out_file.write('] ;\n')
            if parent is not None:
                self.out_file.write('%d -> %d' % (parent, node_id))
                if parent == 0:
                    angles = np.array([45, -45]) * ((self.rotate - 0.5) * -2)
                    self.out_file.write(' [labeldistance=2.5, labelangle=')
                    if node_id == 1:
                        self.out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        self.out_file.write('%d, headlabel="False"]' % angles[1])
                self.out_file.write(' ;\n')
            if left_child != _tree.TREE_LEAF:
                self.recurse(tree, left_child, criterion=criterion, parent=node_id, depth=depth + 1)
                self.recurse(tree, right_child, criterion=criterion, parent=node_id, depth=depth + 1)
        else:
            self.ranks['leaves'].append(str(node_id))
            self.out_file.write('%d [label="(...)"' % node_id)
            if self.filled:
                self.out_file.write(', fillcolor="#C0C0C0"')
            self.out_file.write('] ;\n' % node_id)
            if parent is not None:
                self.out_file.write('%d -> %d ;\n' % (parent, node_id))

class _MPLTreeExporter(_BaseTreeExporter):

    def __init__(self, max_depth=None, feature_names=None, class_names=None, label='all', filled=False, impurity=True, node_ids=False, proportion=False, rounded=False, precision=3, fontsize=None):
        if False:
            return 10
        super().__init__(max_depth=max_depth, feature_names=feature_names, class_names=class_names, label=label, filled=filled, impurity=impurity, node_ids=node_ids, proportion=proportion, rounded=rounded, precision=precision)
        self.fontsize = fontsize
        self.ranks = {'leaves': []}
        self.colors = {'bounds': None}
        self.characters = ['#', '[', ']', '<=', '\n', '', '']
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args['boxstyle'] = 'round'
        self.arrow_args = dict(arrowstyle='<-')

    def _make_tree(self, node_id, et, criterion, depth=0):
        if False:
            for i in range(10):
                print('nop')
        name = self.node_to_str(et, node_id, criterion=criterion)
        if et.children_left[node_id] != _tree.TREE_LEAF and (self.max_depth is None or depth <= self.max_depth):
            children = [self._make_tree(et.children_left[node_id], et, criterion, depth=depth + 1), self._make_tree(et.children_right[node_id], et, criterion, depth=depth + 1)]
        else:
            return Tree(name, node_id)
        return Tree(name, node_id, *children)

    def export(self, decision_tree, ax=None):
        if False:
            for i in range(10):
                print('nop')
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
        draw_tree = buchheim(my_tree)
        (max_x, max_y) = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height
        scale_x = ax_width / max_x
        scale_y = ax_height / max_y
        self.recurse(draw_tree, decision_tree.tree_, ax, max_x, max_y)
        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]
        renderer = ax.figure.canvas.get_renderer()
        for ann in anns:
            ann.update_bbox_position_size(renderer)
        if self.fontsize is None:
            extents = [ann.get_bbox_patch().get_window_extent() for ann in anns]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            size = anns[0].get_fontsize() * min(scale_x / max_width, scale_y / max_height)
            for ann in anns:
                ann.set_fontsize(size)
        return anns

    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        if False:
            return 10
        import matplotlib.pyplot as plt
        kwargs = dict(bbox=self.bbox_args.copy(), ha='center', va='center', zorder=100 - 10 * depth, xycoords='axes fraction', arrowprops=self.arrow_args.copy())
        kwargs['arrowprops']['edgecolor'] = plt.rcParams['text.color']
        if self.fontsize is not None:
            kwargs['fontsize'] = self.fontsize
        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)
        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs['bbox']['fc'] = self.get_fill_color(tree, node.tree.node_id)
            else:
                kwargs['bbox']['fc'] = ax.get_facecolor()
            if node.parent is None:
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = ((node.parent.x + 0.5) / max_x, (max_y - node.parent.y - 0.5) / max_y)
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            for child in node.children:
                self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)
        else:
            xy_parent = ((node.parent.x + 0.5) / max_x, (max_y - node.parent.y - 0.5) / max_y)
            kwargs['bbox']['fc'] = 'grey'
            ax.annotate('\n  (...)  \n', xy_parent, xy, **kwargs)

@validate_params({'decision_tree': 'no_validation', 'out_file': [str, None, HasMethods('write')], 'max_depth': [Interval(Integral, 0, None, closed='left'), None], 'feature_names': ['array-like', None], 'class_names': ['array-like', 'boolean', None], 'label': [StrOptions({'all', 'root', 'none'})], 'filled': ['boolean'], 'leaves_parallel': ['boolean'], 'impurity': ['boolean'], 'node_ids': ['boolean'], 'proportion': ['boolean'], 'rotate': ['boolean'], 'rounded': ['boolean'], 'special_characters': ['boolean'], 'precision': [Interval(Integral, 0, None, closed='left'), None], 'fontname': [str]}, prefer_skip_nested_validation=True)
def export_graphviz(decision_tree, out_file=None, *, max_depth=None, feature_names=None, class_names=None, label='all', filled=False, leaves_parallel=False, impurity=True, node_ids=False, proportion=False, rotate=False, rounded=False, special_characters=False, precision=3, fontname='helvetica'):
    if False:
        return 10
    'Export a decision tree in DOT format.\n\n    This function generates a GraphViz representation of the decision tree,\n    which is then written into `out_file`. Once exported, graphical renderings\n    can be generated using, for example::\n\n        $ dot -Tps tree.dot -o tree.ps      (PostScript format)\n        $ dot -Tpng tree.dot -o tree.png    (PNG format)\n\n    The sample counts that are shown are weighted with any sample_weights that\n    might be present.\n\n    Read more in the :ref:`User Guide <tree>`.\n\n    Parameters\n    ----------\n    decision_tree : object\n        The decision tree estimator to be exported to GraphViz.\n\n    out_file : object or str, default=None\n        Handle or name of the output file. If ``None``, the result is\n        returned as a string.\n\n        .. versionchanged:: 0.20\n            Default of out_file changed from "tree.dot" to None.\n\n    max_depth : int, default=None\n        The maximum depth of the representation. If None, the tree is fully\n        generated.\n\n    feature_names : array-like of shape (n_features,), default=None\n        An array containing the feature names.\n        If None, generic names will be used ("x[0]", "x[1]", ...).\n\n    class_names : array-like of shape (n_classes,) or bool, default=None\n        Names of each of the target classes in ascending numerical order.\n        Only relevant for classification and not supported for multi-output.\n        If ``True``, shows a symbolic representation of the class name.\n\n    label : {\'all\', \'root\', \'none\'}, default=\'all\'\n        Whether to show informative labels for impurity, etc.\n        Options include \'all\' to show at every node, \'root\' to show only at\n        the top root node, or \'none\' to not show at any node.\n\n    filled : bool, default=False\n        When set to ``True``, paint nodes to indicate majority class for\n        classification, extremity of values for regression, or purity of node\n        for multi-output.\n\n    leaves_parallel : bool, default=False\n        When set to ``True``, draw all leaf nodes at the bottom of the tree.\n\n    impurity : bool, default=True\n        When set to ``True``, show the impurity at each node.\n\n    node_ids : bool, default=False\n        When set to ``True``, show the ID number on each node.\n\n    proportion : bool, default=False\n        When set to ``True``, change the display of \'values\' and/or \'samples\'\n        to be proportions and percentages respectively.\n\n    rotate : bool, default=False\n        When set to ``True``, orient tree left to right rather than top-down.\n\n    rounded : bool, default=False\n        When set to ``True``, draw node boxes with rounded corners.\n\n    special_characters : bool, default=False\n        When set to ``False``, ignore special characters for PostScript\n        compatibility.\n\n    precision : int, default=3\n        Number of digits of precision for floating point in the values of\n        impurity, threshold and value attributes of each node.\n\n    fontname : str, default=\'helvetica\'\n        Name of font used to render text.\n\n    Returns\n    -------\n    dot_data : str\n        String representation of the input tree in GraphViz dot format.\n        Only returned if ``out_file`` is None.\n\n        .. versionadded:: 0.18\n\n    Examples\n    --------\n    >>> from sklearn.datasets import load_iris\n    >>> from sklearn import tree\n\n    >>> clf = tree.DecisionTreeClassifier()\n    >>> iris = load_iris()\n\n    >>> clf = clf.fit(iris.data, iris.target)\n    >>> tree.export_graphviz(clf)\n    \'digraph Tree {...\n    '
    if feature_names is not None:
        feature_names = check_array(feature_names, ensure_2d=False, dtype=None, ensure_min_samples=0)
    if class_names is not None and (not isinstance(class_names, bool)):
        class_names = check_array(class_names, ensure_2d=False, dtype=None, ensure_min_samples=0)
    check_is_fitted(decision_tree)
    own_file = False
    return_string = False
    try:
        if isinstance(out_file, str):
            out_file = open(out_file, 'w', encoding='utf-8')
            own_file = True
        if out_file is None:
            return_string = True
            out_file = StringIO()
        exporter = _DOTTreeExporter(out_file=out_file, max_depth=max_depth, feature_names=feature_names, class_names=class_names, label=label, filled=filled, leaves_parallel=leaves_parallel, impurity=impurity, node_ids=node_ids, proportion=proportion, rotate=rotate, rounded=rounded, special_characters=special_characters, precision=precision, fontname=fontname)
        exporter.export(decision_tree)
        if return_string:
            return exporter.out_file.getvalue()
    finally:
        if own_file:
            out_file.close()

def _compute_depth(tree, node):
    if False:
        while True:
            i = 10
    '\n    Returns the depth of the subtree rooted in node.\n    '

    def compute_depth_(current_node, current_depth, children_left, children_right, depths):
        if False:
            for i in range(10):
                print('nop')
        depths += [current_depth]
        left = children_left[current_node]
        right = children_right[current_node]
        if left != -1 and right != -1:
            compute_depth_(left, current_depth + 1, children_left, children_right, depths)
            compute_depth_(right, current_depth + 1, children_left, children_right, depths)
    depths = []
    compute_depth_(node, 1, tree.children_left, tree.children_right, depths)
    return max(depths)

@validate_params({'decision_tree': [DecisionTreeClassifier, DecisionTreeRegressor], 'feature_names': ['array-like', None], 'class_names': ['array-like', None], 'max_depth': [Interval(Integral, 0, None, closed='left'), None], 'spacing': [Interval(Integral, 1, None, closed='left'), None], 'decimals': [Interval(Integral, 0, None, closed='left'), None], 'show_weights': ['boolean']}, prefer_skip_nested_validation=True)
def export_text(decision_tree, *, feature_names=None, class_names=None, max_depth=10, spacing=3, decimals=2, show_weights=False):
    if False:
        i = 10
        return i + 15
    'Build a text report showing the rules of a decision tree.\n\n    Note that backwards compatibility may not be supported.\n\n    Parameters\n    ----------\n    decision_tree : object\n        The decision tree estimator to be exported.\n        It can be an instance of\n        DecisionTreeClassifier or DecisionTreeRegressor.\n\n    feature_names : array-like of shape (n_features,), default=None\n        An array containing the feature names.\n        If None generic names will be used ("feature_0", "feature_1", ...).\n\n    class_names : array-like of shape (n_classes,), default=None\n        Names of each of the target classes in ascending numerical order.\n        Only relevant for classification and not supported for multi-output.\n\n        - if `None`, the class names are delegated to `decision_tree.classes_`;\n        - otherwise, `class_names` will be used as class names instead of\n          `decision_tree.classes_`. The length of `class_names` must match\n          the length of `decision_tree.classes_`.\n\n        .. versionadded:: 1.3\n\n    max_depth : int, default=10\n        Only the first max_depth levels of the tree are exported.\n        Truncated branches will be marked with "...".\n\n    spacing : int, default=3\n        Number of spaces between edges. The higher it is, the wider the result.\n\n    decimals : int, default=2\n        Number of decimal digits to display.\n\n    show_weights : bool, default=False\n        If true the classification weights will be exported on each leaf.\n        The classification weights are the number of samples each class.\n\n    Returns\n    -------\n    report : str\n        Text summary of all the rules in the decision tree.\n\n    Examples\n    --------\n\n    >>> from sklearn.datasets import load_iris\n    >>> from sklearn.tree import DecisionTreeClassifier\n    >>> from sklearn.tree import export_text\n    >>> iris = load_iris()\n    >>> X = iris[\'data\']\n    >>> y = iris[\'target\']\n    >>> decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)\n    >>> decision_tree = decision_tree.fit(X, y)\n    >>> r = export_text(decision_tree, feature_names=iris[\'feature_names\'])\n    >>> print(r)\n    |--- petal width (cm) <= 0.80\n    |   |--- class: 0\n    |--- petal width (cm) >  0.80\n    |   |--- petal width (cm) <= 1.75\n    |   |   |--- class: 1\n    |   |--- petal width (cm) >  1.75\n    |   |   |--- class: 2\n    '
    if feature_names is not None:
        feature_names = check_array(feature_names, ensure_2d=False, dtype=None, ensure_min_samples=0)
    if class_names is not None:
        class_names = check_array(class_names, ensure_2d=False, dtype=None, ensure_min_samples=0)
    check_is_fitted(decision_tree)
    tree_ = decision_tree.tree_
    if is_classifier(decision_tree):
        if class_names is None:
            class_names = decision_tree.classes_
        elif len(class_names) != len(decision_tree.classes_):
            raise ValueError(f'When `class_names` is an array, it should contain as many items as `decision_tree.classes_`. Got {len(class_names)} while the tree was fitted with {len(decision_tree.classes_)} classes.')
    right_child_fmt = '{} {} <= {}\n'
    left_child_fmt = '{} {} >  {}\n'
    truncation_fmt = '{} {}\n'
    if feature_names is not None and len(feature_names) != tree_.n_features:
        raise ValueError('feature_names must contain %d elements, got %d' % (tree_.n_features, len(feature_names)))
    if isinstance(decision_tree, DecisionTreeClassifier):
        value_fmt = '{}{} weights: {}\n'
        if not show_weights:
            value_fmt = '{}{}{}\n'
    else:
        value_fmt = '{}{} value: {}\n'
    if feature_names is not None:
        feature_names_ = [feature_names[i] if i != _tree.TREE_UNDEFINED else None for i in tree_.feature]
    else:
        feature_names_ = ['feature_{}'.format(i) for i in tree_.feature]
    export_text.report = ''

    def _add_leaf(value, class_name, indent):
        if False:
            i = 10
            return i + 15
        val = ''
        is_classification = isinstance(decision_tree, DecisionTreeClassifier)
        if show_weights or not is_classification:
            val = ['{1:.{0}f}, '.format(decimals, v) for v in value]
            val = '[' + ''.join(val)[:-2] + ']'
        if is_classification:
            val += ' class: ' + str(class_name)
        export_text.report += value_fmt.format(indent, '', val)

    def print_tree_recurse(node, depth):
        if False:
            for i in range(10):
                print('nop')
        indent = ('|' + ' ' * spacing) * depth
        indent = indent[:-spacing] + '-' * spacing
        value = None
        if tree_.n_outputs == 1:
            value = tree_.value[node][0]
        else:
            value = tree_.value[node].T[0]
        class_name = np.argmax(value)
        if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
            class_name = class_names[class_name]
        if depth <= max_depth + 1:
            info_fmt = ''
            info_fmt_left = info_fmt
            info_fmt_right = info_fmt
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names_[node]
                threshold = tree_.threshold[node]
                threshold = '{1:.{0}f}'.format(decimals, threshold)
                export_text.report += right_child_fmt.format(indent, name, threshold)
                export_text.report += info_fmt_left
                print_tree_recurse(tree_.children_left[node], depth + 1)
                export_text.report += left_child_fmt.format(indent, name, threshold)
                export_text.report += info_fmt_right
                print_tree_recurse(tree_.children_right[node], depth + 1)
            else:
                _add_leaf(value, class_name, indent)
        else:
            subtree_depth = _compute_depth(tree_, node)
            if subtree_depth == 1:
                _add_leaf(value, class_name, indent)
            else:
                trunc_report = 'truncated branch of depth %d' % subtree_depth
                export_text.report += truncation_fmt.format(indent, trunc_report)
    print_tree_recurse(0, 1)
    return export_text.report