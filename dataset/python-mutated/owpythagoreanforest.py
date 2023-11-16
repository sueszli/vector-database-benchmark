"""Pythagorean forest widget for visualizing random forests."""
from math import log, sqrt
from typing import Any, Callable, Optional
from AnyQt.QtCore import Qt, QRectF, QSize, QPointF, QSizeF, QModelIndex, QItemSelection, QItemSelectionModel, QT_VERSION, QByteArray, QBuffer, QIODevice
from AnyQt.QtGui import QPainter, QPen, QColor, QBrush, QMouseEvent
from AnyQt.QtWidgets import QSizePolicy, QGraphicsScene, QLabel, QSlider, QListView, QStyledItemDelegate, QStyleOptionViewItem, QStyle
from orangewidget.io import PngFormat
from Orange.base import RandomForestModel, TreeModel
from Orange.data import Table
from Orange.widgets import gui, settings
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.pythagorastreeviewer import PythagorasTreeViewer, ContinuousTreeNode
from Orange.widgets.visualize.utils.tree.skltreeadapter import SklTreeAdapter
from Orange.widgets.widget import OWWidget
REPORT_STYLE = '\n<style>\n* {\n  box-sizing: border-box;\n}\n\n.forest_model_row {\n  display: flex;\n  flex-wrap: wrap;\n  padding: 0 4px;\n}\n\n.forest_model_col {\n  flex: 10%;\n  max-width: 10%;\n  padding: 0 4px;\n}\n\n.forest_model_col img {\n  margin-top: 8px;\n  vertical-align: middle;\n}\n\n@media screen and (max-width: 2200px) {\n  .forest_model_col {\n    flex: 25%;\n    max-width: 25%;\n  }\n}\n\n@media screen and (max-width: 1200px) {\n  .forest_model_col {\n    flex: 50%;\n    max-width: 50%;\n  }\n}\n\n@media screen and (max-width: 600px) {\n  .forest_model_col {\n    flex: 100%;\n    max-width: 100%;\n  }\n}\n</style>\n'

class PythagoreanForestModel(PyListModel):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.depth_limit = -1
        self.target_class_idx = None
        self.size_calc_idx = 0
        self.size_adjustment = None
        self.item_scale = 2

    def data(self, index, role=Qt.DisplayRole):
        if False:
            i = 10
            return i + 15
        if not index.isValid():
            return None
        idx = index.row()
        if role == Qt.SizeHintRole:
            return self.item_scale * QSize(100, 100)
        if role == Qt.DisplayRole:
            if 'tree' not in self._other_data[idx]:
                scene = QGraphicsScene(parent=self)
                tree = PythagorasTreeViewer(adapter=self._list[idx], weight_adjustment=OWPythagoreanForest.SIZE_CALCULATION[self.size_calc_idx][1], interactive=False, padding=100, depth_limit=self.depth_limit, target_class_index=self.target_class_idx)
                scene.addItem(tree)
                self._other_data[idx]['scene'] = scene
                self._other_data[idx]['tree'] = tree
            return self._other_data[idx]['scene']
        return super().data(index, role)

    @property
    def trees(self):
        if False:
            i = 10
            return i + 15
        'Get the tree adapters.'
        return self._list

    def update_tree_views(self, func):
        if False:
            while True:
                i = 10
        'Apply `func` to every rendered tree viewer instance.'
        for (idx, tree_data) in enumerate(self._other_data):
            if 'tree' in tree_data:
                func(tree_data['tree'])
                index = self.index(idx)
                if QT_VERSION < 327680:
                    self.dataChanged.emit(index, index)
                else:
                    self.dataChanged.emit(index, index, [Qt.DisplayRole])

    def update_depth(self, depth):
        if False:
            i = 10
            return i + 15
        self.depth_limit = depth
        self.update_tree_views(lambda tree: tree.set_depth_limit(depth))

    def update_target_class(self, idx):
        if False:
            print('Hello World!')
        self.target_class_idx = idx
        self.update_tree_views(lambda tree: tree.target_class_changed(idx))

    def update_item_size(self, scale):
        if False:
            while True:
                i = 10
        self.item_scale = scale / 100
        indices = [idx for (idx, _) in enumerate(self._other_data)]
        self.emitDataChanged(indices)

    def update_size_calc(self, idx):
        if False:
            while True:
                i = 10
        self.size_calc_idx = idx
        (_, size_calc) = OWPythagoreanForest.SIZE_CALCULATION[idx]
        self.update_tree_views(lambda tree: tree.set_size_calc(size_calc))

class PythagorasTreeDelegate(QStyledItemDelegate):

    def paint(self, painter, option, index):
        if False:
            i = 10
            return i + 15
        scene = index.data(Qt.DisplayRole)
        if scene is None:
            super().paint(painter, option, index)
            return
        painter.save()
        rect = QRectF(QPointF(option.rect.topLeft()), QSizeF(option.rect.size()))
        if option.state & QStyle.State_Selected:
            painter.setPen(QPen(QColor(125, 162, 206, 192)))
            painter.setBrush(QBrush(QColor(217, 232, 252, 192)))
        else:
            painter.setPen(QPen(QColor('#ebebeb')))
        painter.drawRoundedRect(rect, 3, 3)
        painter.restore()
        painter.setRenderHint(QPainter.Antialiasing)
        scene.setSceneRect(scene.itemsBoundingRect())
        scene_rect = scene.itemsBoundingRect()
        w_scale = option.rect.width() / scene_rect.width()
        h_scale = option.rect.height() / scene_rect.height()
        scale = min(w_scale, h_scale)
        scene_w = scale * scene_rect.width()
        scene_h = scale * scene_rect.height()
        offset_w = (option.rect.width() - scene_w) / 2
        offset_h = (option.rect.height() - scene_h) / 2
        offset = option.rect.topLeft() + QPointF(offset_w, offset_h)
        target_rect = QRectF(offset, QSizeF(scene_w, scene_h))
        scene.render(painter, target=target_rect, mode=Qt.KeepAspectRatio)

class ClickToClearSelectionListView(QListView):
    """Clicking outside any item clears the current selection."""

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        super().mousePressEvent(event)
        index = self.indexAt(event.pos())
        if index.row() == -1:
            self.clearSelection()

class OWPythagoreanForest(OWWidget):
    name = 'Pythagorean Forest'
    description = 'Pythagorean forest for visualising random forests.'
    icon = 'icons/PythagoreanForest.svg'
    settings_version = 2
    keywords = 'pythagorean forest, fractal'
    priority = 1001

    class Inputs:
        random_forest = Input('Random Forest', RandomForestModel, replaces=['Random forest'])

    class Outputs:
        tree = Output('Tree', TreeModel)
    settingsHandler = settings.ClassValuesContextHandler()
    depth_limit = settings.Setting(10)
    target_class_index = settings.ContextSetting(0)
    size_calc_idx = settings.Setting(0)
    zoom = settings.Setting(200)
    selected_index = settings.ContextSetting(None)
    SIZE_CALCULATION = [('Normal', lambda x: x), ('Square root', lambda x: sqrt(x)), ('Logarithmic', lambda x: log(x + 1))]

    @classmethod
    def migrate_settings(cls, settings, version):
        if False:
            for i in range(10):
                print('nop')
        if version < 2:
            settings.pop('selected_tree_index', None)
            (v1_min, v1_max) = (20, 150)
            (v2_min, v2_max) = (100, 400)
            ratio = (v2_max - v2_min) / (v1_max - v1_min)
            settings['zoom'] = int(ratio * (settings['zoom'] - v1_min) + v2_min)

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.rf_model = None
        self.forest = None
        self.instances = None
        self.color_palette = None
        box_info = gui.widgetBox(self.controlArea, 'Forest')
        self.ui_info = gui.widgetLabel(box_info)
        box_display = gui.widgetBox(self.controlArea, 'Display')
        self.ui_depth_slider = gui.hSlider(box_display, self, 'depth_limit', label='Depth', ticks=False, maxValue=900)
        self.ui_target_class_combo = gui.comboBox(box_display, self, 'target_class_index', label='Target class', orientation=Qt.Horizontal, items=[], contentsLength=8, searchable=True)
        self.ui_size_calc_combo = gui.comboBox(box_display, self, 'size_calc_idx', label='Size', orientation=Qt.Horizontal, items=list(zip(*self.SIZE_CALCULATION))[0], contentsLength=8)
        self.ui_zoom_slider = gui.hSlider(box_display, self, 'zoom', label='Zoom', ticks=False, minValue=100, maxValue=400, createLabel=False, intOnly=False)
        gui.rubber(self.controlArea)
        self.controlArea.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.forest_model = PythagoreanForestModel(parent=self)
        self.forest_model.update_item_size(self.zoom)
        self.ui_depth_slider.valueChanged.connect(self.forest_model.update_depth)
        self.ui_target_class_combo.currentIndexChanged.connect(self.forest_model.update_target_class)
        self.ui_zoom_slider.valueChanged.connect(self.forest_model.update_item_size)
        self.ui_size_calc_combo.currentIndexChanged.connect(self.forest_model.update_size_calc)
        self.list_delegate = PythagorasTreeDelegate(parent=self)
        self.list_view = ClickToClearSelectionListView(parent=self)
        self.list_view.setWrapping(True)
        self.list_view.setFlow(QListView.LeftToRight)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setModel(self.forest_model)
        self.list_view.setItemDelegate(self.list_delegate)
        self.list_view.setSpacing(2)
        self.list_view.setSelectionMode(QListView.SingleSelection)
        self.list_view.selectionModel().selectionChanged.connect(self.commit)
        self.list_view.setUniformItemSizes(True)
        self.mainArea.layout().addWidget(self.list_view)
        self.resize(800, 500)
        self.clear()

    @Inputs.random_forest
    def set_rf(self, model=None):
        if False:
            print('Hello World!')
        'When a different forest is given.'
        self.closeContext()
        self.clear()
        self.rf_model = model
        if model is not None:
            self.instances = model.instances
            self._update_target_class_combo()
            self.forest = self._get_forest_adapter(self.rf_model)
            self.forest_model[:] = self.forest.trees
            self._update_info_box()
            self._update_depth_slider()
            self.openContext(model.domain.class_var if model.domain is not None else None)
        if self.selected_index is not None:
            index = self.list_view.model().index(self.selected_index)
            selection = QItemSelection(index, index)
            self.list_view.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)

    def clear(self):
        if False:
            print('Hello World!')
        'Clear all relevant data from the widget.'
        self.rf_model = None
        self.forest = None
        self.forest_model.clear()
        self.selected_index = None
        self._clear_info_box()
        self._clear_target_class_combo()
        self._clear_depth_slider()

    def _update_info_box(self):
        if False:
            return 10
        self.ui_info.setText('Trees: {}'.format(len(self.forest.trees)))

    def _update_depth_slider(self):
        if False:
            print('Hello World!')
        self.depth_limit = self._get_max_depth()
        self.ui_depth_slider.parent().setEnabled(True)
        self.ui_depth_slider.setMaximum(self.depth_limit)
        self.ui_depth_slider.setValue(self.depth_limit)

    def _update_target_class_combo(self):
        if False:
            i = 10
            return i + 15
        self._clear_target_class_combo()
        label = [x for x in self.ui_target_class_combo.parent().children() if isinstance(x, QLabel)][0]
        if self.instances.domain.has_discrete_class:
            label_text = 'Target class'
            values = [c.title() for c in self.instances.domain.class_vars[0].values]
            values.insert(0, 'None')
        else:
            label_text = 'Node color'
            values = list(ContinuousTreeNode.COLOR_METHODS.keys())
        label.setText(label_text)
        self.ui_target_class_combo.addItems(values)
        self.target_class_index = 0

    def _clear_info_box(self):
        if False:
            i = 10
            return i + 15
        self.ui_info.setText('No forest on input.')

    def _clear_target_class_combo(self):
        if False:
            return 10
        self.ui_target_class_combo.clear()
        self.target_class_index = -1

    def _clear_depth_slider(self):
        if False:
            while True:
                i = 10
        self.ui_depth_slider.parent().setEnabled(False)
        self.ui_depth_slider.setMaximum(0)

    def _get_max_depth(self):
        if False:
            for i in range(10):
                print('nop')
        return max((tree.max_depth for tree in self.forest.trees))

    def _get_forest_adapter(self, model):
        if False:
            while True:
                i = 10
        return SklRandomForestAdapter(model)

    def onDeleteWidget(self):
        if False:
            return 10
        'When deleting the widget.'
        super().onDeleteWidget()
        self.clear()

    def commit(self, selection: QItemSelection) -> None:
        if False:
            while True:
                i = 10
        'Commit the selected tree to output.'
        selected_indices = selection.indexes()
        if not len(selected_indices):
            self.selected_index = None
            self.Outputs.tree.send(None)
            return
        self.selected_index = selected_indices[0].row()
        tree = self.rf_model.trees[self.selected_index]
        tree.instances = self.instances
        tree.meta_target_class_index = self.target_class_index
        tree.meta_size_calc_idx = self.size_calc_idx
        tree.meta_depth_limit = self.depth_limit
        self.Outputs.tree.send(tree)

    def send_report(self):
        if False:
            i = 10
            return i + 15
        'Send report.'
        model = self.forest_model
        max_rows = 30

        def item_html(row):
            if False:
                return 10
            img_data = model.data(model.index(row))
            byte_array = QByteArray()
            filename = QBuffer(byte_array)
            filename.open(QIODevice.WriteOnly)
            PngFormat.write(filename, img_data)
            img_encoded = byte_array.toBase64().data().decode('utf-8')
            return f'<img style="width:100%" src="data:image/png;base64,{img_encoded}"/>'
        html = ["<div class='forest_model_row'>"]
        for i in range(model.rowCount())[:max_rows]:
            html.append("<div class='forest_model_col'>")
            html.extend(item_html(i))
            html.append('</div>')
        html.append('</div>')
        html = REPORT_STYLE + ''.join(html)
        if model.rowCount() > max_rows:
            html += '<p>. . .</p>'
        self.report_raw(html)

class SklRandomForestAdapter:
    """Take a `RandomForest` and wrap all the trees into the `SklTreeAdapter`
    instances that Pythagorean trees use."""

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        self._adapters = None
        self._domain = model.domain
        self._trees = model.trees

    @property
    def trees(self):
        if False:
            print('Hello World!')
        'Get the tree adapters in the random forest.'
        if not self._adapters:
            self._adapters = list(map(SklTreeAdapter, self._trees))
        return self._adapters

    @property
    def domain(self):
        if False:
            while True:
                i = 10
        'Get the domain.'
        return self._domain
if __name__ == '__main__':
    from Orange.modelling import RandomForestLearner
    data = Table('iris')
    rf = RandomForestLearner(n_estimators=10)(data)
    rf.instances = data
    WidgetPreview(OWPythagoreanForest).run(rf)