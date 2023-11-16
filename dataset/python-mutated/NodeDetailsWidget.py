"""
DEV - NOT USED SO FAR
"""
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QGroupBox
from qtpy.QtGui import QFont

class NodeDetailsWidget(QGroupBox):

    def __init__(self, main_window, nodes_tree_widget):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('details')
        self.main_window = main_window
        self.nodes_tree_widget = nodes_tree_widget
        self.nodes_tree_widget.details_widget = self
        self.setLayout(QVBoxLayout())
        self.doc_text_edit = QTextEdit()
        self.doc_text_edit.setFont(QFont('Consolas', 9))
        self.doc_text_edit.setReadOnly(True)
        self.layout().addWidget(self.doc_text_edit)
        self.node = None

    def set_node(self, node):
        if False:
            i = 10
            return i + 15
        self.node = node
        self.update_details()

    def update_details(self):
        if False:
            while True:
                i = 10
        self.doc_text_edit.setText(self.node.doc)