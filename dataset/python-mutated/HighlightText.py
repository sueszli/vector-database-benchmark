"""
Created on 2019年5月22日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file:
@description:
"""
import sys
try:
    from PyQt5.QtCore import QRegExp
    from PyQt5.QtGui import QTextCharFormat, QTextCursor
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QToolBar, QLineEdit, QPushButton, QColorDialog, QHBoxLayout, QWidget
except ImportError:
    from PySide2.QtCore import QRegExp
    from PySide2.QtGui import QTextCharFormat, QTextCursor
    from PySide2.QtWidgets import QApplication, QMainWindow, QTextEdit, QToolBar, QLineEdit, QPushButton, QColorDialog, QHBoxLayout, QWidget

class TextEdit(QMainWindow):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super(TextEdit, self).__init__(parent)
        self.textEdit = QTextEdit(self)
        self.setCentralWidget(self.textEdit)
        widget = QWidget(self)
        vb = QHBoxLayout(widget)
        vb.setContentsMargins(0, 0, 0, 0)
        self.findText = QLineEdit(self)
        self.findText.setText('self')
        findBtn = QPushButton('高亮', self)
        findBtn.clicked.connect(self.highlight)
        vb.addWidget(self.findText)
        vb.addWidget(findBtn)
        tb = QToolBar(self)
        tb.addWidget(widget)
        self.addToolBar(tb)

    def setText(self, text):
        if False:
            while True:
                i = 10
        self.textEdit.setPlainText(text)

    def highlight(self):
        if False:
            while True:
                i = 10
        text = self.findText.text()
        if not text:
            return
        col = QColorDialog.getColor(self.textEdit.textColor(), self)
        if not col.isValid():
            return
        cursor = self.textEdit.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.setCharFormat(QTextCharFormat())
        cursor.clearSelection()
        self.textEdit.setTextCursor(cursor)
        fmt = QTextCharFormat()
        fmt.setForeground(col)
        expression = QRegExp(text)
        self.textEdit.moveCursor(QTextCursor.Start)
        cursor = self.textEdit.textCursor()
        pos = 0
        index = expression.indexIn(self.textEdit.toPlainText(), pos)
        while index >= 0:
            cursor.setPosition(index)
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, len(text))
            cursor.mergeCharFormat(fmt)
            pos = index + expression.matchedLength()
            index = expression.indexIn(self.textEdit.toPlainText(), pos)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    textEdit = TextEdit()
    textEdit.resize(800, 600)
    textEdit.show()
    textEdit.setText(open(sys.argv[0], 'rb').read().decode())
    sys.exit(app.exec_())