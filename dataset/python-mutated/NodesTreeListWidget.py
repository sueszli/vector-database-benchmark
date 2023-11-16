"""
DEV - NOT USED SO FAR
"""
from qtpy.QtCore import QMimeData, QByteArray, Qt, QPoint, Signal
from qtpy.QtGui import QStandardItemModel, QStandardItem, QDrag
from qtpy.QtWidgets import QWidget, QTreeView, QVBoxLayout, QMenu, QAction

class NodesView(QTreeView):
    node_selected = Signal(object)

    def mousePressEvent(self, event) -> None:
        if False:
            return 10
        item = self.model().itemFromIndex(self.indexAt(event.pos()))
        if isinstance(item, NodeItem):
            self.node_selected.emit(item.node)
        super().mousePressEvent(event)

class NodeItem(QStandardItem):

    def __init__(self, node):
        if False:
            print('Hello World!')
        super().__init__(node.title)
        self.node = node
        self.setEditable(False)
        self.setDragEnabled(True)

class NodesTreeListWidget(QWidget):

    def __init__(self, main_window, session):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.main_window = main_window
        self.session = session
        self.details_widget = None
        self.view = NodesView()
        self.view.node_selected.connect(self._node_selected)
        self.model = QStandardItemModel()
        self.view.setModel(self.model)
        self.view.setHeaderHidden(True)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.view)
        self.setMaximumWidth(500)

    def update_list(self):
        if False:
            return 10
        self.model.clear()
        packages = {}
        for (node, node_package) in self.main_window.node_packages.items():
            if node_package in packages.keys():
                packages[node_package].append(node)
            else:
                packages[node_package] = [node]
        for (node_package, nodes) in packages.items():
            package_item = QStandardItem(node_package.name)
            package_item.setEditable(False)
            for n in nodes:
                node_item = NodeItem(n)
                package_item.appendRow(node_item)
            self.model.appendRow(package_item)

    def _node_selected(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.details_widget.set_node(node)