import json
from qtpy.QtWidgets import QLineEdit, QWidget, QLabel, QGridLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QStyleOption, QStyle
from qtpy.QtGui import QFont, QPainter, QColor, QDrag
from qtpy.QtCore import Signal, Qt, QMimeData

class NodeWidget(QWidget):
    chosen = Signal()
    custom_focused_from_inside = Signal()

    def __init__(self, parent, node):
        if False:
            while True:
                i = 10
        super(NodeWidget, self).__init__(parent)
        self.custom_focused = False
        self.node = node
        self.left_mouse_pressed_on_me = False
        main_layout = QGridLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self_ = self

        class NameLabel(QLineEdit):

            def __init__(self, text):
                if False:
                    while True:
                        i = 10
                super().__init__(text)
                self.setReadOnly(True)
                self.setFont(QFont('Source Code Pro', 8))

            def mouseMoveEvent(self, ev):
                if False:
                    while True:
                        i = 10
                self_.custom_focused_from_inside.emit()
                ev.ignore()

            def mousePressEvent(self, ev):
                if False:
                    i = 10
                    return i + 15
                ev.ignore()

            def mouseReleaseEvent(self, ev):
                if False:
                    while True:
                        i = 10
                ev.ignore()
        name_label = NameLabel(node.title)
        type_layout = QHBoxLayout()
        main_layout.addWidget(name_label, 0, 0)
        self.setLayout(main_layout)
        self.setContentsMargins(0, 0, 0, 0)
        self.setMaximumWidth(250)
        self.setToolTip(node.__doc__)
        self.update_stylesheet()

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        self.custom_focused_from_inside.emit()
        if event.button() == Qt.LeftButton:
            self.left_mouse_pressed_on_me = True

    def mouseMoveEvent(self, event):
        if False:
            i = 10
            return i + 15
        if self.left_mouse_pressed_on_me:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setData('application/json', bytes(json.dumps({'type': 'node', 'node identifier': self.node.identifier}), encoding='utf-8'))
            drag.setMimeData(mime_data)
            drop_action = drag.exec_()

    def mouseReleaseEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.left_mouse_pressed_on_me = False
        if self.geometry().contains(self.mapToParent(event.pos())):
            self.chosen.emit()

    def set_custom_focus(self, new_focus):
        if False:
            for i in range(10):
                print('nop')
        self.custom_focused = new_focus
        self.update_stylesheet()

    def update_stylesheet(self):
        if False:
            while True:
                i = 10
        color = self.node.GUI.color if hasattr(self.node, 'GUI') else '#888888'
        (r, g, b) = (QColor(color).red(), QColor(color).green(), QColor(color).blue())
        new_style_sheet = f"\nNodeWidget {{\n    border: 1px solid rgba(255,255,255,150);\n    border-radius: 2px;\n    {(f'background-color: rgba(255,255,255,80);' if self.custom_focused else '')}\n}}\nQLabel {{\n    background: transparent;\n}}\nQLineEdit {{\n    color: white;\n    background: transparent;\n    border: none;\n    padding: 2px;\n}}\n        "
        self.setStyleSheet(new_style_sheet)

    def paintEvent(self, event):
        if False:
            while True:
                i = 10
        o = QStyleOption()
        o.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, o, p, self)