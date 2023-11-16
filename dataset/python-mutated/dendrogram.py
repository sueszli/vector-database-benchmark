import fractions
from collections import namedtuple, OrderedDict
from typing import Tuple, List, Optional, Dict
import numpy as np
from AnyQt.QtCore import QPointF, QRectF, Qt, QSizeF, QEvent, Signal
from AnyQt.QtGui import QPainterPath, QPen, QBrush, QPalette, QPainterPathStroker, QColor, QTransform, QFontMetrics, QPolygonF
from AnyQt.QtWidgets import QGraphicsWidget, QGraphicsPathItem, QGraphicsItemGroup, QGraphicsSimpleTextItem
from Orange.clustering.hierarchical import Tree, postorder, preorder, leaves
from Orange.widgets.utils import colorpalettes
__all__ = ['DendrogramWidget']

def dendrogram_layout(tree, expand_leaves=False):
    if False:
        i = 10
        return i + 15
    coords = []
    cluster_geometry = {}
    leaf_idx = 0
    for node in postorder(tree):
        cluster = node.value
        if node.is_leaf:
            if expand_leaves:
                start = float(cluster.first) + 0.5
                end = float(cluster.last - 1) + 0.5
            else:
                start = end = leaf_idx + 0.5
                leaf_idx += 1
            center = (start + end) / 2.0
            cluster_geometry[node] = (start, center, end)
            coords.append((node, (start, center, end)))
        else:
            left = node.left
            right = node.right
            left_center = cluster_geometry[left][1]
            right_center = cluster_geometry[right][1]
            (start, end) = (left_center, right_center)
            center = (start + end) / 2.0
            cluster_geometry[node] = (start, center, end)
            coords.append((node, (start, center, end)))
    return coords
Point = namedtuple('Point', ['x', 'y'])
Element = namedtuple('Element', ['anchor', 'path'])

def path_toQtPath(geom):
    if False:
        while True:
            i = 10
    p = QPainterPath()
    (anchor, points) = geom
    if len(points) > 1:
        p.moveTo(*points[0])
        for (x, y) in points[1:]:
            p.lineTo(x, y)
    elif len(points) == 1:
        r = QRectF(0, 0, 1.0, 1e-09)
        r.moveCenter(*points[0])
        p.addRect(r)
    elif len(points) == 0:
        r = QRectF(0, 0, 1e-16, 1e-16)
        r.moveCenter(QPointF(*anchor))
        p.addRect(r)
    return p
(Left, Top, Right, Bottom) = (1, 2, 3, 4)

def dendrogram_path(tree, orientation=Left, scaleh=1):
    if False:
        return 10
    layout = dendrogram_layout(tree)
    T = {}
    paths = {}
    rootdata = tree.value
    base = scaleh * rootdata.height
    if orientation == Bottom:
        transform = lambda x, y: (x, y)
    if orientation == Top:
        transform = lambda x, y: (x, base - y)
    elif orientation == Left:
        transform = lambda x, y: (base - y, x)
    elif orientation == Right:
        transform = lambda x, y: (y, x)
    for (node, (start, center, end)) in layout:
        if node.is_leaf:
            (x, y) = transform(center, 0)
            anchor = Point(x, y)
            paths[node] = Element(anchor, ())
        else:
            (left, right) = (paths[node.left], paths[node.right])
            lines = (left.anchor, Point(*transform(start, scaleh * node.value.height)), Point(*transform(end, scaleh * node.value.height)), right.anchor)
            anchor = Point(*transform(center, scaleh * node.value.height))
            paths[node] = Element(anchor, lines)
        T[node] = Tree((node, paths[node]), tuple((T[ch] for ch in node.branches)))
    return T[tree]

def make_pen(brush=Qt.black, width=1, style=Qt.SolidLine, cap_style=Qt.SquareCap, join_style=Qt.BevelJoin, cosmetic=False):
    if False:
        print('Hello World!')
    pen = QPen(brush)
    pen.setWidth(width)
    pen.setStyle(style)
    pen.setCapStyle(cap_style)
    pen.setJoinStyle(join_style)
    pen.setCosmetic(cosmetic)
    return pen

def update_pen(pen, brush=None, width=None, style=None, cap_style=None, join_style=None, cosmetic=None):
    if False:
        while True:
            i = 10
    pen = QPen(pen)
    if brush is not None:
        pen.setBrush(QBrush(brush))
    if width is not None:
        pen.setWidth(width)
    if style is not None:
        pen.setStyle(style)
    if cap_style is not None:
        pen.setCapStyle(cap_style)
    if join_style is not None:
        pen.setJoinStyle(join_style)
    if cosmetic is not None:
        pen.setCosmetic(cosmetic)
    return pen

def path_stroke(path, width=1, join_style=Qt.RoundJoin):
    if False:
        i = 10
        return i + 15
    stroke = QPainterPathStroker()
    stroke.setWidth(width)
    stroke.setJoinStyle(join_style)
    stroke.setMiterLimit(1.0)
    return stroke.createStroke(path)

def path_outline(path, width=1, join_style=Qt.RoundJoin):
    if False:
        for i in range(10):
            print('nop')
    stroke = path_stroke(path, width, join_style)
    return stroke.united(path)

class DendrogramWidget(QGraphicsWidget):
    """A Graphics Widget displaying a dendrogram."""

    class ClusterGraphicsItem(QGraphicsPathItem):
        sourcePath = QPainterPath()
        sourceAreaShape = QPainterPath()
        __shape = None
        __boundingRect = None
        __mouseAreaShape = QPainterPath()

        def setGeometryData(self, path, hitArea):
            if False:
                i = 10
                return i + 15
            '\n            Set the geometry (path) and the mouse hit area (hitArea) for this\n            item.\n            '
            super().setPath(path)
            self.prepareGeometryChange()
            self.__boundingRect = self.__shape = None
            self.__mouseAreaShape = hitArea

        def shape(self):
            if False:
                for i in range(10):
                    print('nop')
            if self.__shape is None:
                path = super().shape()
                self.__shape = path.united(self.__mouseAreaShape)
            return self.__shape

        def boundingRect(self):
            if False:
                i = 10
                return i + 15
            if self.__boundingRect is None:
                sh = self.shape()
                pw = self.pen().widthF() / 2.0
                self.__boundingRect = sh.boundingRect().adjusted(-pw, -pw, pw, pw)
            return self.__boundingRect

    class _SelectionItem(QGraphicsItemGroup):

        def __init__(self, parent, path, unscaled_path, label=''):
            if False:
                return 10
            super().__init__(parent)
            self.path = QGraphicsPathItem(path, self)
            self.path.setPen(make_pen(width=1, cosmetic=True))
            self.addToGroup(self.path)
            self.label = QGraphicsSimpleTextItem(label)
            self._update_label_pos()
            self.addToGroup(self.label)
            self.unscaled_path = unscaled_path

        def set_path(self, path):
            if False:
                print('Hello World!')
            self.path.setPath(path)
            self._update_label_pos()

        def set_label(self, label):
            if False:
                while True:
                    i = 10
            self.label.setText(label)
            self._update_label_pos()

        def set_color(self, color):
            if False:
                i = 10
                return i + 15
            self.path.setBrush(QColor(color))

        def _update_label_pos(self):
            if False:
                i = 10
                return i + 15
            path = self.path.path()
            elements = (path.elementAt(i) for i in range(path.elementCount()))
            points = ((p.x, p.y) for p in elements)
            (p1, p2, *rest) = sorted(points)
            (x, y) = (p1[0], (p1[1] + p2[1]) / 2)
            brect = self.label.boundingRect()
            self.label.setPos(x - brect.width() - 4, y - brect.height() + 4 * (len(rest) == 3))
    (Left, Top, Right, Bottom) = (1, 2, 3, 4)
    (NoSelection, SingleSelection, ExtendedSelection) = (0, 1, 2)
    itemClicked = Signal(ClusterGraphicsItem)
    selectionChanged = Signal()
    selectionEdited = Signal()

    def __init__(self, parent=None, root=None, orientation=Left, hoverHighlightEnabled=True, selectionMode=ExtendedSelection, *, pen_width=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(None, **kwargs)
        self.setFiltersChildEvents(True)
        self.orientation = orientation
        self._root = None
        self._layout = None
        self._highlighted_item = None
        self._selection = OrderedDict()
        self._items = {}
        self._itemgroup = QGraphicsWidget(self)
        self._itemgroup.setGeometry(self.contentsRect())
        self._transform = QTransform()
        self._cluster_parent = {}
        self.__hoverHighlightEnabled = hoverHighlightEnabled
        self.__selectionMode = selectionMode
        self._pen_width = pen_width
        self.setContentsMargins(0, 0, 0, 0)
        self.setRoot(root)
        if parent is not None:
            self.setParentItem(parent)

    def setSelectionMode(self, mode):
        if False:
            print('Hello World!')
        '\n        Set the selection mode.\n        '
        assert mode in [DendrogramWidget.NoSelection, DendrogramWidget.SingleSelection, DendrogramWidget.ExtendedSelection]
        if self.__selectionMode != mode:
            self.__selectionMode = mode
            if self.__selectionMode == DendrogramWidget.NoSelection and self._selection:
                self.setSelectedClusters([])
            elif self.__selectionMode == DendrogramWidget.SingleSelection and len(self._selection) > 1:
                self.setSelectedClusters([self.selected_nodes()[-1]])

    def selectionMode(self):
        if False:
            while True:
                i = 10
        '\n        Return the current selection mode.\n        '
        return self.__selectionMode

    def setHoverHighlightEnabled(self, enabled):
        if False:
            print('Hello World!')
        if self.__hoverHighlightEnabled != bool(enabled):
            self.__hoverHighlightEnabled = bool(enabled)
            if self._highlighted_item is not None:
                self._set_hover_item(None)

    def isHoverHighlightEnabled(self):
        if False:
            i = 10
            return i + 15
        return self.__hoverHighlightEnabled

    def clear(self):
        if False:
            print('Hello World!')
        '\n        Clear the widget.\n        '
        scene = self.scene()
        if scene is not None:
            scene.removeItem(self._itemgroup)
        else:
            self._itemgroup.setParentItem(None)
        self._itemgroup = QGraphicsWidget(self)
        self._itemgroup.setGeometry(self.contentsRect())
        self._items.clear()
        for item in self._selection.values():
            if scene is not None:
                scene.removeItem(item)
            else:
                item.setParentItem(None)
        self._root = None
        self._items = {}
        self._selection = OrderedDict()
        self._highlighted_item = None
        self._cluster_parent = {}
        self.updateGeometry()

    def setRoot(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the root cluster tree node for display.\n\n        Parameters\n        ----------\n        root : Tree\n            The tree root node.\n        '
        self.clear()
        self._root = root
        if root is not None:
            foreground = self.palette().color(QPalette.WindowText)
            pen = make_pen(foreground, width=self._pen_width, cosmetic=True)
            for node in postorder(root):
                item = DendrogramWidget.ClusterGraphicsItem(self._itemgroup)
                item.setAcceptHoverEvents(True)
                item.setPen(pen)
                item.node = node
                for branch in node.branches:
                    assert branch in self._items
                    self._cluster_parent[branch] = node
                self._items[node] = item
            self._relayout()
            self._rescale()
        self.updateGeometry()
    set_root = setRoot

    def root(self):
        if False:
            while True:
                i = 10
        '\n        Return the cluster tree root node.\n\n        Returns\n        -------\n        root : Tree\n        '
        return self._root

    def item(self, node):
        if False:
            return 10
        '\n        Return the ClusterGraphicsItem instance representing the cluster `node`.\n        '
        return self._items.get(node)

    def heightAt(self, point):
        if False:
            i = 10
            return i + 15
        '\n        Return the cluster height at the point in widget local coordinates.\n        '
        if not self._root:
            return 0
        (tinv, ok) = self._transform.inverted()
        if not ok:
            return 0
        tpoint = tinv.map(point)
        if self.orientation in [self.Left, self.Right]:
            height = tpoint.x()
        else:
            height = tpoint.y()
        base = self._root.value.height
        scale = self._height_scale_factor()
        Fr = fractions.Fraction
        if scale > 0:
            height = Fr(height) / Fr(scale)
        else:
            height = 0
        if self.orientation in [self.Left, self.Bottom]:
            height = Fr(base) - Fr(height)
        return float(height)
    height_at = heightAt

    def posAtHeight(self, height):
        if False:
            while True:
                i = 10
        '\n        Return a point in local coordinates for `height` (in cluster\n        '
        if not self._root:
            return QPointF()
        scale = self._height_scale_factor()
        base = self._root.value.height
        height = scale * height
        if self.orientation in [self.Left, self.Bottom]:
            height = scale * base - height
        if self.orientation in [self.Left, self.Right]:
            p = QPointF(height, 0)
        else:
            p = QPointF(0, height)
        return self._transform.map(p)
    pos_at_height = posAtHeight

    def _set_hover_item(self, item):
        if False:
            print('Hello World!')
        'Set the currently highlighted item.'
        if self._highlighted_item is item:
            return

        def set_pen(item, pen):
            if False:
                for i in range(10):
                    print('nop')

            def branches(item):
                if False:
                    for i in range(10):
                        print('nop')
                return [self._items[ch] for ch in item.node.branches]
            for it in postorder(item, branches):
                it.setPen(pen)
        if self._highlighted_item:
            highlight = self.palette().color(QPalette.WindowText)
            set_pen(self._highlighted_item, make_pen(highlight, width=self._pen_width, cosmetic=True))
        self._highlighted_item = item
        if item:
            hpen = make_pen(self.palette().color(QPalette.Highlight), width=self._pen_width + 1, cosmetic=True)
            set_pen(item, hpen)

    def leafItems(self):
        if False:
            while True:
                i = 10
        'Iterate over the dendrogram leaf items (:class:`QGraphicsItem`).\n        '
        if self._root:
            return (self._items[leaf] for leaf in leaves(self._root))
        else:
            return iter(())
    leaf_items = leafItems

    def leafAnchors(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterate over the dendrogram leaf anchor points (:class:`QPointF`).\n\n        The points are in the widget local coordinates.\n        '
        for item in self.leafItems():
            anchor = QPointF(item.element.anchor)
            yield self.mapFromItem(item, anchor)
    leaf_anchors = leafAnchors

    def selectedNodes(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the selected cluster nodes.\n        '
        return [item.node for item in self._selection]
    selected_nodes = selectedNodes

    def setSelectedItems(self, items: List[ClusterGraphicsItem]):
        if False:
            print('Hello World!')
        'Set the item selection.'
        to_remove = set(self._selection) - set(items)
        to_add = set(items) - set(self._selection)
        for sel in to_remove:
            self._remove_selection(sel)
        for sel in to_add:
            self._add_selection(sel)
        if to_add or to_remove:
            self._re_enumerate_selections()
            self.selectionChanged.emit()
    set_selected_items = setSelectedItems

    def setSelectedClusters(self, clusters: List[Tree]) -> None:
        if False:
            while True:
                i = 10
        'Set the selected clusters.\n        '
        self.setSelectedItems(list(map(self.item, clusters)))
    set_selected_clusters = setSelectedClusters

    def isItemSelected(self, item: ClusterGraphicsItem) -> bool:
        if False:
            while True:
                i = 10
        'Is `item` selected (is a root of a selection).'
        return item in self._selection

    def isItemIncludedInSelection(self, item: ClusterGraphicsItem) -> bool:
        if False:
            return 10
        'Is item included in any selection.'
        return self._selected_super_item(item) is not None
    is_included = isItemIncludedInSelection

    def setItemSelected(self, item, state):
        if False:
            return 10
        'Set the `item`s selection state to `state`.'
        if state is False and item not in self._selection or (state is True and item in self._selection):
            return
        if item in self._selection:
            if state is False:
                self._remove_selection(item)
                self._re_enumerate_selections()
                self.selectionChanged.emit()
        else:
            super_selection = self._selected_super_item(item)
            if super_selection:
                self._remove_selection(super_selection)
            sub_selections = self._selected_sub_items(item)
            for sub in sub_selections:
                self._remove_selection(sub)
            if state:
                self._add_selection(item)
            elif item in self._selection:
                self._remove_selection(item)
            self._re_enumerate_selections()
            self.selectionChanged.emit()
    select_item = setItemSelected

    @staticmethod
    def _create_path(item, path):
        if False:
            while True:
                i = 10
        ppath = QPainterPath()
        if item.node.is_leaf:
            ppath.addRect(path.boundingRect().adjusted(-8, -4, 0, 4))
        else:
            ppath.addPolygon(path)
            ppath = path_outline(ppath, width=-8)
        return ppath

    @staticmethod
    def _create_label(i):
        if False:
            print('Hello World!')
        return f'C{i + 1}'

    def _add_selection(self, item):
        if False:
            return 10
        'Add selection rooted at item\n        '
        outline = self._selection_poly(item)
        path = self._transform.map(outline)
        ppath = self._create_path(item, path)
        label = self._create_label(len(self._selection))
        selection_item = self._SelectionItem(self, ppath, outline, label)
        selection_item.label.setBrush(self.palette().color(QPalette.Link))
        selection_item.setPos(self.contentsRect().topLeft())
        self._selection[item] = selection_item

    def _remove_selection(self, item):
        if False:
            return 10
        'Remove selection rooted at item.'
        selection_item = self._selection[item]
        selection_item.hide()
        selection_item.setParentItem(None)
        if self.scene():
            self.scene().removeItem(selection_item)
        del self._selection[item]

    def _selected_sub_items(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Return all selected subclusters under item.'

        def branches(item):
            if False:
                return 10
            return [self._items[ch] for ch in item.node.branches]
        res = []
        for item in list(preorder(item, branches))[1:]:
            if item in self._selection:
                res.append(item)
        return res

    def _selected_super_item(self, item):
        if False:
            i = 10
            return i + 15
        'Return the selected super item if it exists.'

        def branches(item):
            if False:
                i = 10
                return i + 15
            return [self._items[ch] for ch in item.node.branches]
        for selected_item in self._selection:
            if item in set(preorder(selected_item, branches)):
                return selected_item
        return None

    def _re_enumerate_selections(self):
        if False:
            i = 10
            return i + 15
        'Re enumerate the selection items and update the colors.'
        items = sorted(self._selection.items(), key=lambda item: item[0].node.value.first)
        palette = colorpalettes.LimitedDiscretePalette(len(items))
        for (i, (item, selection_item)) in enumerate(items):
            del self._selection[item]
            self._selection[item] = selection_item
            selection_item.set_label(self._create_label(i))
            color = palette[i]
            color.setAlpha(150)
            selection_item.set_color(color)

    def _selection_poly(self, item):
        if False:
            while True:
                i = 10
        '\n        Return an selection geometry covering item and all its children.\n        '

        def left(item):
            if False:
                for i in range(10):
                    print('nop')
            return [self._items[ch] for ch in item.node.branches[:1]]

        def right(item):
            if False:
                i = 10
                return i + 15
            return [self._items[ch] for ch in item.node.branches[-1:]]
        itemsleft = list(preorder(item, left))[::-1]
        itemsright = list(preorder(item, right))
        assert itemsleft[0].node.is_leaf
        assert itemsright[-1].node.is_leaf
        if item.node.is_leaf:
            vert = [itemsleft[0].element.anchor]
        else:
            vert = []
            for it in itemsleft[1:]:
                vert.extend([it.element.path[0], it.element.path[1], it.element.anchor])
            for it in itemsright[:-1]:
                vert.extend([it.element.anchor, it.element.path[-2], it.element.path[-1]])
            vert.append(vert[0])

            def isclose(a, b, rel_tol=1e-06):
                if False:
                    while True:
                        i = 10
                return abs(a - b) < rel_tol * max(abs(a), abs(b))

            def isclose_p(p1, p2, rel_tol=1e-06):
                if False:
                    print('Hello World!')
                return isclose(p1.x, p2.x, rel_tol) and isclose(p1.y, p2.y, rel_tol)
            acc = [vert[0]]
            for v in vert[1:]:
                if not isclose_p(v, acc[-1]):
                    acc.append(v)
            vert = acc
        return QPolygonF([QPointF(*p) for p in vert])

    def _update_selection_items(self):
        if False:
            return 10
        'Update the shapes of selection items after a scale change.\n        '
        transform = self._transform
        for (item, selection) in self._selection.items():
            path = transform.map(selection.unscaled_path)
            ppath = self._create_path(item, path)
            selection.set_path(ppath)

    def _height_scale_factor(self):
        if False:
            i = 10
            return i + 15
        if self._root is None:
            return 1
        base = self._root.value.height
        if base >= np.finfo(base).eps:
            return 1 / base
        else:
            return 0

    def _relayout(self):
        if False:
            i = 10
            return i + 15
        if self._root is None:
            return
        scale = self._height_scale_factor()
        base = scale * self._root.value.height
        self._layout = dendrogram_path(self._root, self.orientation, scaleh=scale)
        for node_geom in postorder(self._layout):
            (node, geom) = node_geom.value
            item = self._items[node]
            item.element = geom
            item.sourcePath = path_toQtPath(geom)
            r = item.sourcePath.boundingRect()
            if self.orientation == Left:
                r.setRight(base)
            elif self.orientation == Right:
                r.setLeft(0)
            elif self.orientation == Top:
                r.setBottom(base)
            else:
                r.setTop(0)
            hitarea = QPainterPath()
            hitarea.addRect(r)
            item.sourceAreaShape = hitarea
            item.setGeometryData(item.sourcePath, item.sourceAreaShape)
            item.setZValue(-node.value.height)

    def _rescale(self):
        if False:
            for i in range(10):
                print('nop')
        if self._root is None:
            return
        scale = self._height_scale_factor()
        base = scale * self._root.value.height
        crect = self.contentsRect()
        leaf_count = len(list(leaves(self._root)))
        if self.orientation in [Left, Right]:
            drect = QSizeF(base, leaf_count)
        else:
            drect = QSizeF(leaf_count, base)
        eps = np.finfo(np.float64).eps
        if abs(drect.width()) < eps:
            sx = 1.0
        else:
            sx = crect.width() / drect.width()
        if abs(drect.height()) < eps:
            sy = 1.0
        else:
            sy = crect.height() / drect.height()
        transform = QTransform().scale(sx, sy)
        self._transform = transform
        self._itemgroup.setPos(crect.topLeft())
        self._itemgroup.setGeometry(crect)
        for node_geom in postorder(self._layout):
            (node, _) = node_geom.value
            item = self._items[node]
            item.setGeometryData(transform.map(item.sourcePath), transform.map(item.sourceAreaShape))
        self._selection_items = None
        self._update_selection_items()

    def sizeHint(self, which: Qt.SizeHint, constraint=QSizeF()) -> QSizeF:
        if False:
            while True:
                i = 10
        fm = QFontMetrics(self.font())
        spacing = fm.lineSpacing()
        (mleft, mtop, mright, mbottom) = self.getContentsMargins()
        if self._root and which == Qt.PreferredSize:
            nleaves = len([node for node in self._items.keys() if not node.branches])
            base = max(10, min(spacing * 16, 250))
            if self.orientation in [self.Left, self.Right]:
                return QSizeF(base, spacing * nleaves + mleft + mright)
            else:
                return QSizeF(spacing * nleaves + mtop + mbottom, base)
        elif which == Qt.MinimumSize:
            return QSizeF(mleft + mright + 10, mtop + mbottom + 10)
        else:
            return QSizeF()

    def sceneEventFilter(self, obj, event):
        if False:
            i = 10
            return i + 15
        if isinstance(obj, DendrogramWidget.ClusterGraphicsItem):
            if event.type() == QEvent.GraphicsSceneHoverEnter and self.__hoverHighlightEnabled:
                self._set_hover_item(obj)
                event.accept()
                return True
            elif event.type() == QEvent.GraphicsSceneMousePress and event.button() == Qt.LeftButton:
                is_selected = self.isItemSelected(obj)
                is_included = self.is_included(obj)
                current_selection = list(self._selection)
                if self.__selectionMode == DendrogramWidget.SingleSelection:
                    if event.modifiers() & Qt.ControlModifier:
                        self.setSelectedItems([obj] if not is_selected else [])
                    elif event.modifiers() & Qt.AltModifier:
                        self.setSelectedItems([])
                    elif event.modifiers() & Qt.ShiftModifier:
                        if not is_included:
                            self.setSelectedItems([obj])
                    elif current_selection != [obj]:
                        self.setSelectedItems([obj])
                elif self.__selectionMode == DendrogramWidget.ExtendedSelection:
                    if event.modifiers() & Qt.ControlModifier:
                        self.setItemSelected(obj, not is_selected)
                    elif event.modifiers() & Qt.AltModifier:
                        self.setItemSelected(self._selected_super_item(obj), False)
                    elif event.modifiers() & Qt.ShiftModifier:
                        if not is_included:
                            self.setItemSelected(obj, True)
                    elif current_selection != [obj]:
                        self.setSelectedItems([obj])
                if current_selection != self._selection:
                    self.selectionEdited.emit()
                self.itemClicked.emit(obj)
                event.accept()
                return True
        if event.type() == QEvent.GraphicsSceneHoverLeave:
            self._set_hover_item(None)
        return super().sceneEventFilter(obj, event)

    def changeEvent(self, event):
        if False:
            while True:
                i = 10
        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            self.updateGeometry()
        elif event.type() == QEvent.PaletteChange:
            self._update_colors()
        elif event.type() == QEvent.ContentsRectChange:
            self._rescale()

    def resizeEvent(self, event):
        if False:
            i = 10
            return i + 15
        super().resizeEvent(event)
        self._rescale()

    def mousePressEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super().mousePressEvent(event)
        if event.modifiers() == Qt.NoModifier and self._selection:
            self.set_selected_clusters([])

    def _update_colors(self):
        if False:
            i = 10
            return i + 15

        def set_color(item: DendrogramWidget.ClusterGraphicsItem, color: QColor):
            if False:
                i = 10
                return i + 15

            def branches(item):
                if False:
                    print('Hello World!')
                return [self._items[ch] for ch in item.node.branches]
            for it in postorder(item, branches):
                it.setPen(update_pen(it.pen(), brush=color))
        if self._root is not None:
            foreground = self.palette().color(QPalette.WindowText)
            item = self.item(self._root)
            set_color(item, foreground)
        highlight = self.palette().color(QPalette.Highlight)
        if self._highlighted_item is not None:
            set_color(self._highlighted_item, highlight)
        accent = self.palette().color(QPalette.Link)
        for item in self._selection.values():
            item.label.setBrush(accent)