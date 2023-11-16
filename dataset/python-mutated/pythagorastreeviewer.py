"""
Pythagoras tree viewer for visualizing tree structures.

The pythagoras tree viewer widget is a widget that can be plugged into any
existing widget given a tree adapter instance. It is simply a canvas that takes
and input tree adapter and takes care of all the drawing.

Types
-----
Square : namedtuple (center, length, angle)
    Since Pythagoras trees deal only with squares (they also deal with
    rectangles in the generalized form, but are completely unreadable), this
    is what all the squares are stored as.
Point : namedtuple (x, y)
    Self exaplanatory.

"""
from abc import ABCMeta, abstractmethod
from collections import namedtuple, defaultdict, deque
from math import pi, sqrt, cos, sin, degrees
import numpy as np
from AnyQt.QtCore import Qt, QTimer, QRectF, QSizeF
from AnyQt.QtGui import QColor, QPen
from AnyQt.QtWidgets import QSizePolicy, QGraphicsItem, QGraphicsRectItem, QGraphicsWidget, QStyle
from Orange.widgets.utils import to_html
from Orange.widgets.visualize.utils.tree.rules import Rule
from Orange.widgets.visualize.utils.tree.treeadapter import TreeAdapter
Z_STEP = 5000000
Square = namedtuple('Square', ['center', 'length', 'angle'])
Point = namedtuple('Point', ['x', 'y'])

class PythagorasTreeViewer(QGraphicsWidget):
    """Pythagoras tree viewer graphics widget.

    Examples
    --------
    >>> from Orange.widgets.visualize.utils.tree.treeadapter import (
    ...     TreeAdapter
    ... )
    Pass tree through constructor.
    >>> tree_view = PythagorasTreeViewer(parent=scene, adapter=tree_adapter)

    Pass tree later through method.
    >>> tree_adapter = TreeAdapter()
    >>> scene = QGraphicsScene()
    This is where the magic happens
    >>> tree_view = PythagorasTreeViewer(parent=scene)
    >>> tree_view.set_tree(tree_adapter)

    Both these examples set the appropriate tree and add all the squares to the
    widget instance.

    Parameters
    ----------
    parent : QGraphicsItem, optional
        The parent object that the graphics widget belongs to. Should be a
        scene.
    adapter : TreeAdapter, optional
        Any valid tree adapter instance.
    interacitive : bool, optional
        Specify whether the widget should have an interactive display. This
        means special hover effects, selectable boxes. Default is true.

    Notes
    -----
    .. note:: The class contains two clear methods: `clear` and `clear_tree`.
        Each has  their own use.
        `clear_tree` will clear out the tree and remove any graphics items.
        `clear` will, on the other hand, clear everything, all settings
        (tooltip and color calculation functions.

        This is useful because when we want to change the size calculation of
        the Pythagora tree, we just want to clear the scene and it would be
        inconvenient to have to set color and tooltip functions again.
        On the other hand, when we want to draw a brand new tree, it is best
        to clear all settings to avoid any strange bugs - we start with a blank
        slate.

    """

    def __init__(self, parent=None, adapter=None, depth_limit=0, padding=0, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.parent = parent
        self.tree_adapter = None
        self.root = None
        self._depth_limit = depth_limit
        self._interactive = kwargs.get('interactive', True)
        self._padding = padding
        self._square_objects = {}
        self._drawn_nodes = deque()
        self._frontier = deque()
        self._target_class_index = 0
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if adapter is not None:
            self.set_tree(adapter, target_class_index=kwargs.get('target_class_index'), weight_adjustment=kwargs.get('weight_adjustment'))
            self.set_depth_limit(depth_limit)

    def set_tree(self, tree_adapter, weight_adjustment=lambda x: x, target_class_index=0):
        if False:
            i = 10
            return i + 15
        'Pass in a new tree adapter instance and perform updates to canvas.\n\n        Parameters\n        ----------\n        tree_adapter : TreeAdapter\n            The new tree adapter that is to be used.\n        weight_adjustment : callable\n            A weight adjustment function that with signature `x -> x`\n        target_class_index : int\n\n        Returns\n        -------\n\n        '
        self.clear_tree()
        self.tree_adapter = tree_adapter
        self.weight_adjustment = weight_adjustment
        if self.tree_adapter is not None:
            self.root = self._calculate_tree(self.tree_adapter, self.weight_adjustment)
            self.set_depth_limit(tree_adapter.max_depth)
            self.target_class_changed(target_class_index)
            self._draw_tree(self.root)

    def set_size_calc(self, weight_adjustment):
        if False:
            while True:
                i = 10
        'Set the weight adjustment on the tree. Redraws the whole tree.'
        self.weight_adjustment = weight_adjustment
        self.set_tree(self.tree_adapter, self.weight_adjustment, self._target_class_index)

    def set_depth_limit(self, depth):
        if False:
            for i in range(10):
                print('nop')
        'Update the drawing depth limit.\n\n        The drawing stops when the depth is GT the limit. This means that at\n        depth 0, the root node will be drawn.\n\n        Parameters\n        ----------\n        depth : int\n            The maximum depth at which the nodes can still be drawn.\n\n        Returns\n        -------\n\n        '
        self._depth_limit = depth
        self._draw_tree(self.root)

    def target_class_changed(self, target_class_index=0):
        if False:
            for i in range(10):
                print('nop')
        'When the target class has changed, perform appropriate updates.'
        self._target_class_index = target_class_index

        def _recurse(node):
            if False:
                for i in range(10):
                    print('nop')
            node.target_class_index = target_class_index
            for child in node.children:
                _recurse(child)
        _recurse(self.root)

    def tooltip_changed(self, tooltip_enabled):
        if False:
            return 10
        'Set the tooltip to the appropriate value on each square.'
        for square in self._squares():
            if tooltip_enabled:
                square.setToolTip(square.tree_node.tooltip)
            else:
                square.setToolTip(None)

    def clear(self):
        if False:
            return 10
        'Clear the entire widget state.'
        self.clear_tree()
        self._target_class_index = 0

    def clear_tree(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear only the tree, keeping tooltip and color functions.'
        self.tree_adapter = None
        self.root = None
        self._clear_scene()

    def _calculate_tree(self, tree_adapter, weight_adjustment):
        if False:
            i = 10
            return i + 15
        'Actually calculate the tree squares'
        tree_builder = PythagorasTree(weight_adjustment=weight_adjustment)
        return tree_builder.pythagoras_tree(tree_adapter, tree_adapter.root, Square(Point(0, 0), 200, -pi / 2))

    def _draw_tree(self, root):
        if False:
            i = 10
            return i + 15
        "Efficiently draw the tree with regards to the depth.\n\n        If we used a recursive approach, the tree would have to be redrawn\n        every time the depth changed, which is very impractical for larger\n        trees, since drawing can take a long time.\n\n        Using an iterative approach, we use two queues to represent the tree\n        frontier and the nodes that have already been drawn. We also store the\n        current depth. This way, when the max depth is increased, we do not\n        redraw the entire tree but only iterate through the frontier and draw\n        those nodes, and update the frontier accordingly.\n        When decreasing the max depth, we reverse the process, we clear the\n        frontier, and remove nodes from the drawn nodes, and append those with\n        depth max_depth + 1 to the frontier, so the frontier doesn't get\n        cluttered.\n\n        Parameters\n        ----------\n        root : TreeNode\n            The root tree node.\n\n        Returns\n        -------\n\n        "
        if self.root is None:
            return
        if not self._drawn_nodes:
            self._frontier.appendleft((0, root))
        was_decreased = self._depth_was_decreased()
        if was_decreased:
            self._frontier.clear()
        while self._drawn_nodes:
            (depth, node) = self._drawn_nodes.pop()
            if depth <= self._depth_limit:
                self._drawn_nodes.append((depth, node))
                break
            if depth == self._depth_limit + 1:
                self._frontier.appendleft((depth, node))
            if node.label in self._square_objects:
                self._square_objects[node.label].hide()
        while self._frontier:
            (depth, node) = self._frontier.popleft()
            if depth > self._depth_limit:
                self._frontier.appendleft((depth, node))
                break
            self._drawn_nodes.append((depth, node))
            self._frontier.extend(((depth + 1, c) for c in node.children))
            node.target_class_index = self._target_class_index
            if node.label in self._square_objects:
                self._square_objects[node.label].show()
            else:
                square_obj = InteractiveSquareGraphicsItem if self._interactive else SquareGraphicsItem
                self._square_objects[node.label] = square_obj(node, parent=self, zvalue=depth)

    def _depth_was_decreased(self):
        if False:
            i = 10
            return i + 15
        if not self._drawn_nodes:
            return False
        (depth, node) = self._drawn_nodes.pop()
        self._drawn_nodes.append((depth, node))
        return depth > self._depth_limit

    def _squares(self):
        if False:
            i = 10
            return i + 15
        return [node.graphics_item for (_, node) in self._drawn_nodes]

    def _clear_scene(self):
        if False:
            return 10
        for square in self._squares():
            self.scene().removeItem(square)
        self._frontier.clear()
        self._drawn_nodes.clear()
        self._square_objects.clear()

    def boundingRect(self):
        if False:
            for i in range(10):
                print('nop')
        return self.childrenBoundingRect().adjusted(-self._padding, -self._padding, self._padding, self._padding)

    def sizeHint(self, size_hint, size_constraint=None, *args, **kwargs):
        if False:
            return 10
        return self.boundingRect().size() + QSizeF(self._padding, self._padding)

class SquareGraphicsItem(QGraphicsRectItem):
    """Square Graphics Item.

    Square component to draw as components for the non-interactive Pythagoras
    tree.

    Parameters
    ----------
    tree_node : TreeNode
        The tree node the square represents.
    parent : QGraphicsItem

    """

    def __init__(self, tree_node, parent=None, **kwargs):
        if False:
            while True:
                i = 10
        self.tree_node = tree_node
        super().__init__(self._get_rect_attributes(), parent)
        self.tree_node.graphics_item = self
        self.setTransformOriginPoint(self.boundingRect().center())
        self.setRotation(degrees(self.tree_node.square.angle))
        self.setBrush(kwargs.get('brush', QColor('#297A1F')))
        pen = QPen(QColor(Qt.black))
        pen.setWidthF(0.75)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setAcceptHoverEvents(True)
        self.setZValue(kwargs.get('zvalue', 0))
        self.z_step = Z_STEP
        if self.tree_node.parent != TreeAdapter.ROOT_PARENT:
            p = self.tree_node.parent
            num_children = len(p.children)
            own_index = [1 if c.label == self.tree_node.label else 0 for c in p.children].index(1)
            self.z_step = int(p.graphics_item.z_step / num_children)
            base_z = p.graphics_item.zValue()
            self.setZValue(base_z + own_index * self.z_step)

    def update(self):
        if False:
            print('Hello World!')
        self.setBrush(self.tree_node.color)
        return super().update()

    def _get_rect_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the rectangle attributes requrired to draw item.\n\n        Compute the QRectF that a QGraphicsRect needs to be rendered with the\n        data passed down in the constructor.\n\n        '
        (center, length, _) = self.tree_node.square
        x = center[0] - length / 2
        y = center[1] - length / 2
        return QRectF(x, y, length, length)

class InteractiveSquareGraphicsItem(SquareGraphicsItem):
    """Interactive square graphics items.

    This is different from the base square graphics item so that it is
    selectable, and it can handle and react to hover events (highlight and
    focus own branch).

    Parameters
    ----------
    tree_node : TreeNode
        The tree node the square represents.
    parent : QGraphicsItem

    """
    timer = QTimer()
    MAX_OPACITY = 1.0
    SELECTION_OPACITY = 0.5
    HOVER_OPACITY = 0.1

    def __init__(self, tree_node, parent=None, **kwargs):
        if False:
            return 10
        super().__init__(tree_node, parent, **kwargs)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.initial_zvalue = self.zValue()
        self.any_selected = False
        self.timer.setSingleShot(True)

    def update(self):
        if False:
            return 10
        self.setToolTip(self.tree_node.tooltip)
        return super().update()

    def hoverEnterEvent(self, event):
        if False:
            print('Hello World!')
        self.timer.stop()

        def fnc(graphics_item):
            if False:
                while True:
                    i = 10
            graphics_item.setZValue(Z_STEP)
            if self.any_selected:
                if graphics_item.isSelected():
                    opacity = self.MAX_OPACITY
                else:
                    opacity = self.SELECTION_OPACITY
            else:
                opacity = self.MAX_OPACITY
            graphics_item.setOpacity(opacity)

        def other_fnc(graphics_item):
            if False:
                return 10
            if graphics_item.isSelected():
                opacity = self.MAX_OPACITY
            else:
                opacity = self.HOVER_OPACITY
            graphics_item.setOpacity(opacity)
            graphics_item.setZValue(self.initial_zvalue)
        self._propagate_z_values(self, fnc, other_fnc)

    def hoverLeaveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')

        def fnc(graphics_item):
            if False:
                while True:
                    i = 10
            graphics_item.setZValue(self.initial_zvalue)

        def other_fnc(graphics_item):
            if False:
                while True:
                    i = 10
            if self.any_selected:
                if graphics_item.isSelected():
                    opacity = self.MAX_OPACITY
                else:
                    opacity = self.SELECTION_OPACITY
            else:
                opacity = self.MAX_OPACITY
            graphics_item.setOpacity(opacity)
        self.timer.timeout.connect(lambda : self._propagate_z_values(self, fnc, other_fnc))
        self.timer.start(250)

    def _propagate_z_values(self, graphics_item, fnc, other_fnc):
        if False:
            i = 10
            return i + 15
        self._propagate_to_children(graphics_item, fnc)
        self._propagate_to_parents(graphics_item, fnc, other_fnc)

    def _propagate_to_children(self, graphics_item, fnc):
        if False:
            print('Hello World!')
        fnc(graphics_item)
        for c in graphics_item.tree_node.children:
            self._propagate_to_children(c.graphics_item, fnc)

    def _propagate_to_parents(self, graphics_item, fnc, other_fnc):
        if False:
            i = 10
            return i + 15
        if graphics_item.tree_node.parent != TreeAdapter.ROOT_PARENT:
            parent = graphics_item.tree_node.parent.graphics_item
            for c in parent.tree_node.children:
                if c != graphics_item.tree_node:
                    self._propagate_to_children(c.graphics_item, other_fnc)
            fnc(parent)
            self._propagate_to_parents(parent, fnc, other_fnc)

    def mouseDoubleClickEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.tree_node.tree.reverse_children(self.tree_node.label)
        p = self.parentWidget()
        p.set_tree(p.tree_adapter, p.weight_adjustment, self.tree_node.target_class_index)
        widget = p.parent
        widget._update_main_area()

    def selection_changed(self):
        if False:
            i = 10
            return i + 15
        'Handle selection changed.'
        self.any_selected = len(self.scene().selectedItems()) > 0
        if self.any_selected:
            if self.isSelected():
                self.setOpacity(self.MAX_OPACITY)
            elif self.opacity() != self.HOVER_OPACITY:
                self.setOpacity(self.SELECTION_OPACITY)
        else:
            self.setGraphicsEffect(None)
            self.setOpacity(self.MAX_OPACITY)

    def paint(self, painter, option, widget=None):
        if False:
            while True:
                i = 10
        if self.isSelected():
            option.state ^= QStyle.State_Selected
            rect = self.rect()
            super().paint(painter, option, widget)
            painter.save()
            pen = QPen(QColor(Qt.black))
            pen.setWidthF(2)
            pen.setCosmetic(True)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.restore()
        else:
            super().paint(painter, option, widget)

class TreeNode(metaclass=ABCMeta):
    """A tree node meant to be used in conjuction with graphics items.

    The tree node contains methods that are very general to any tree
    visualisation, containing methods for the node color and tooltip.

    This is an abstract class and not meant to be used by itself. There are two
    subclasses - `DiscreteTreeNode` and `ContinuousTreeNode`, which need no
    explanation. If you don't wish to deal with figuring out which node to use,
    the `from_tree` method is provided.

    Parameters
    ----------
    label : int
        The label of the tree node, can be looked up in the original tree.
    square : Square
        The square the represents the tree node.
    tree : TreeAdapter
        The tree model that the node belongs to.
    children : tuple of TreeNode, optional, default is empty tuple
        All the children that belong to this node.

    """

    def __init__(self, label, square, tree, children=()):
        if False:
            while True:
                i = 10
        self.label = label
        self.square = square
        self.tree = tree
        self.children = children
        self.parent = None
        self.__graphics_item = None
        self.__target_class_index = None

    @property
    def graphics_item(self):
        if False:
            return 10
        return self.__graphics_item

    @graphics_item.setter
    def graphics_item(self, graphics_item):
        if False:
            for i in range(10):
                print('nop')
        self.__graphics_item = graphics_item
        self._update_graphics_item()

    @property
    def target_class_index(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__target_class_index

    @target_class_index.setter
    def target_class_index(self, target_class_index):
        if False:
            i = 10
            return i + 15
        self.__target_class_index = target_class_index
        self._update_graphics_item()

    def _update_graphics_item(self):
        if False:
            return 10
        if self.__graphics_item is not None:
            self.__graphics_item.update()

    @classmethod
    def from_tree(cls, label, square, tree, children=()):
        if False:
            print('Hello World!')
        'Construct the appropriate type of node from the given tree.'
        if tree.domain.has_discrete_class:
            node = DiscreteTreeNode
        else:
            node = ContinuousTreeNode
        return node(label, square, tree, children)

    @property
    @abstractmethod
    def color(self):
        if False:
            print('Hello World!')
        'Get the color of the node.\n\n        Returns\n        -------\n        QColor\n\n        '

    @property
    @abstractmethod
    def tooltip(self):
        if False:
            print('Hello World!')
        'get the tooltip for the node.\n\n        Returns\n        -------\n        str\n\n        '

    @property
    def color_palette(self):
        if False:
            return 10
        return self.tree.domain.class_var.palette

    def _rules_str(self):
        if False:
            return 10
        rules = self.tree.rules(self.label)
        if rules:
            if isinstance(rules[0], Rule):
                sorted_rules = sorted(rules[:-1], key=lambda rule: rule.attr_name)
                return '<br>'.join((str(rule) for rule in sorted_rules)) + '<br><b>%s</b>' % rules[-1]
            else:
                return '<br>'.join((to_html(rule) for rule in rules))
        else:
            return ''

class DiscreteTreeNode(TreeNode):
    """Discrete tree node containing methods for tree visualisations.

    Colors are defined by the data domain, and possible colorings are different
    target classes.

    """

    @property
    def color(self):
        if False:
            return 10
        distribution = self.tree.get_distribution(self.label)[0]
        total = np.sum(distribution)
        if self.target_class_index:
            p = distribution[self.target_class_index - 1] / total
            color = self.color_palette[self.target_class_index - 1]
            color = color.lighter(int(200 - 100 * p))
        else:
            modus = np.argmax(distribution)
            p = distribution[modus] / (total or 1)
            color = self.color_palette[int(modus)]
            color = color.lighter(int(400 - 300 * p))
        return color

    @property
    def tooltip(self):
        if False:
            print('Hello World!')
        distribution = self.tree.get_distribution(self.label)[0]
        total = int(np.sum(distribution))
        if self.target_class_index:
            samples = distribution[self.target_class_index - 1]
            text = ''
        else:
            modus = np.argmax(distribution)
            samples = distribution[modus]
            text = self.tree.domain.class_vars[0].values[modus] + '<br>'
        ratio = samples / np.sum(distribution)
        rules_str = self._rules_str()
        splitting_attr = self.tree.attribute(self.label)
        return '<p>' + text + '{}/{} samples ({:2.3f}%)'.format(int(samples), total, ratio * 100) + '<hr>' + ('Split by ' + splitting_attr.name if not self.tree.is_leaf(self.label) else '') + ('<br><br>' if rules_str and (not self.tree.is_leaf(self.label)) else '') + rules_str + '</p>'

class ContinuousTreeNode(TreeNode):
    """Continuous tree node containing methods for tree visualisations.

    There are three modes of coloring:
     - None, which is a solid color
     - Mean, which colors nodes w.r.t. the mean value of all the
       instances that belong to a given node.
     - Standard deviation, which colors nodes w.r.t the standard deviation of
       all the instances that belong to a given node.

    """
    (COLOR_NONE, COLOR_MEAN, COLOR_STD) = range(3)
    COLOR_METHODS = {'None': COLOR_NONE, 'Mean': COLOR_MEAN, 'Standard deviation': COLOR_STD}

    @property
    def color(self):
        if False:
            for i in range(10):
                print('nop')
        if self.target_class_index is self.COLOR_MEAN:
            return self._color_mean()
        elif self.target_class_index is self.COLOR_STD:
            return self._color_var()
        else:
            return QColor(255, 255, 255)

    def _color_mean(self):
        if False:
            i = 10
            return i + 15
        'Color the nodes with respect to the mean of instances inside.'
        min_mean = np.min(self.tree.instances.Y)
        max_mean = np.max(self.tree.instances.Y)
        instances = self.tree.get_instances_in_nodes(self.label)
        mean = np.mean(instances.Y)
        return self.color_palette.value_to_qcolor(mean, low=min_mean, high=max_mean)

    def _color_var(self):
        if False:
            i = 10
            return i + 15
        'Color the nodes with respect to the variance of instances inside.'
        (min_std, max_std) = (0, np.std(self.tree.instances.Y))
        instances = self.tree.get_instances_in_nodes(self.label)
        std = np.std(instances.Y)
        return self.color_palette.value_to_qcolor(std, low=min_std, high=max_std)

    @property
    def tooltip(self):
        if False:
            for i in range(10):
                print('nop')
        num_samples = self.tree.num_samples(self.label)
        instances = self.tree.get_instances_in_nodes(self.label)
        mean = np.mean(instances.Y)
        std = np.std(instances.Y)
        rules_str = self._rules_str()
        splitting_attr = self.tree.attribute(self.label)
        return '<p>Mean: {:2.3f}'.format(mean) + '<br>Standard deviation: {:2.3f}'.format(std) + '<br>{} samples'.format(num_samples) + '<hr>' + ('Split by ' + splitting_attr.name if not self.tree.is_leaf(self.label) else '') + ('<br><br>' if rules_str and (not self.tree.is_leaf(self.label)) else '') + rules_str + '</p>'

class PythagorasTree:
    """Pythagoras tree.

    Contains all the logic that converts a given tree adapter to a tree
    consisting of node classes.

    Parameters
    ----------
    weight_adjustment : callable
        The function to be used to adjust child weights

    """

    def __init__(self, weight_adjustment=lambda x: x):
        if False:
            for i in range(10):
                print('nop')
        self.adjust_weight = weight_adjustment
        self._slopes = defaultdict(list)

    def pythagoras_tree(self, tree, node, square):
        if False:
            print('Hello World!')
        'Get the Pythagoras tree representation in a graph like view.\n\n        Constructs a graph using TreeNode into a tree structure. Each node in\n        graph contains the information required to plot the the tree.\n\n        Parameters\n        ----------\n        tree : TreeAdapter\n            A tree adapter instance where the original tree is stored.\n        node : int\n            The node label, the root node is denoted with 0.\n        square : Square\n            The initial square which will represent the root of the tree.\n\n        Returns\n        -------\n        TreeNode\n            The root node which contains the rest of the tree.\n\n        '
        if node == tree.root:
            self._slopes.clear()
        child_weights = [self.adjust_weight(tree.weight(c)) for c in tree.children(node)]
        total_weight = sum(child_weights)
        normalized_child_weights = [cw / total_weight for cw in child_weights]
        children = tuple((self._compute_child(tree, square, child, cw) for (child, cw) in zip(tree.children(node), normalized_child_weights)))
        obj = TreeNode.from_tree(node, square, tree, children)
        for c in children:
            c.parent = obj
        return obj

    def _compute_child(self, tree, parent_square, node, weight):
        if False:
            while True:
                i = 10
        'Compute all the properties for a single child.\n\n        Parameters\n        ----------\n        tree : TreeAdapter\n            A tree adapter instance where the original tree is stored.\n        parent_square : Square\n            The parent square of the given child.\n        node : int\n            The node label of the child.\n        weight : float\n            The weight of the node relative to its parent e.g. two children in\n            relation 3:1 should have weights .75 and .25, respectively.\n\n        Returns\n        -------\n        TreeNode\n            The tree node representation of the given child with the computed\n            subtree.\n\n        '
        alpha = weight * pi
        length = parent_square.length * sin(alpha / 2)
        prev_angles = sum(self._slopes[parent_square])
        center = self._compute_center(parent_square, length, alpha, prev_angles)
        angle = parent_square.angle - pi / 2 + prev_angles + alpha / 2
        square = Square(center, length, angle)
        self._slopes[parent_square].append(alpha)
        return self.pythagoras_tree(tree, node, square)

    def _compute_center(self, initial_square, length, alpha, base_angle=0):
        if False:
            i = 10
            return i + 15
        'Compute the central point of a child square.\n\n        Parameters\n        ----------\n        initial_square : Square\n            The parent square representation where we will be drawing from.\n        length : float\n            The length of the side of the new square (the one we are computing\n            the center for).\n        alpha : float\n            The angle that defines the size of our new square (in radians).\n        base_angle : float, optional\n            If the square we want to find the center for is not the first child\n            i.e. its edges does not touch the base square, then we need the\n            initial angle that will act as the starting point for the new\n            square.\n\n        Returns\n        -------\n        Point\n            The central point to the new square.\n\n        '
        (parent_center, parent_length, parent_angle) = initial_square
        t0 = self._get_point_on_square_edge(parent_center, parent_length, parent_angle)
        square_diagonal_length = sqrt(2 * parent_length ** 2)
        edge = self._get_point_on_square_edge(parent_center, square_diagonal_length, parent_angle - pi / 4)
        if base_angle != 0:
            edge = self._rotate_point(edge, t0, base_angle)
        t1 = self._rotate_point(edge, t0, alpha)
        t2 = Point((t1.x + edge.x) / 2, (t1.y + edge.y) / 2)
        slope = parent_angle - pi / 2 + alpha / 2
        return self._get_point_on_square_edge(t2, length, slope + base_angle)

    @staticmethod
    def _rotate_point(point, around, alpha):
        if False:
            print('Hello World!')
        'Rotate a point around another point by some angle.\n\n        Parameters\n        ----------\n        point : Point\n            The point to rotate.\n        around : Point\n            The point to perform rotation around.\n        alpha : float\n            The angle to rotate by (in radians).\n\n        Returns\n        -------\n        Point:\n            The rotated point.\n\n        '
        temp = Point(point.x - around.x, point.y - around.y)
        temp = Point(temp.x * cos(alpha) - temp.y * sin(alpha), temp.x * sin(alpha) + temp.y * cos(alpha))
        return Point(temp.x + around.x, temp.y + around.y)

    @staticmethod
    def _get_point_on_square_edge(center, length, angle):
        if False:
            print('Hello World!')
        'Calculate the central point on the drawing edge of the given square.\n\n        Parameters\n        ----------\n        center : Point\n            The square center point.\n        length : float\n            The square side length.\n        angle : float\n            The angle of the square.\n\n        Returns\n        -------\n        Point\n            A point on the center of the drawing edge of the given square.\n\n        '
        return Point(center.x + length / 2 * cos(angle), center.y + length / 2 * sin(angle))