from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPalette
from PyQt5.QtWidgets import QToolButton, QLineEdit, QStyle, QStyleOptionFrame
from hscommon.trans import trget
tr = trget('ui')

class LineEditButton(QToolButton):

    def __init__(self, parent, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(parent, **kwargs)
        pixmap = QPixmap(':/search_clear_13')
        self.setIcon(QIcon(pixmap))
        self.setIconSize(pixmap.size())
        self.setCursor(Qt.ArrowCursor)
        self.setPopupMode(QToolButton.InstantPopup)
        stylesheet = 'QToolButton { border: none; padding: 0px; }'
        self.setStyleSheet(stylesheet)

class ClearableEdit(QLineEdit):

    def __init__(self, parent=None, is_clearable=True, **kwargs):
        if False:
            return 10
        super().__init__(parent, **kwargs)
        self._is_clearable = is_clearable
        if is_clearable:
            self._clearButton = LineEditButton(self)
            frame_width = self.style().pixelMetric(QStyle.PM_DefaultFrameWidth)
            padding_right = self._clearButton.sizeHint().width() + frame_width + 1
            stylesheet = f'QLineEdit {{ padding-right:{padding_right}px; }}'
            self.setStyleSheet(stylesheet)
            self._updateClearButton()
            self._clearButton.clicked.connect(self._clearSearch)
        self.textChanged.connect(self._textChanged)

    def _clearSearch(self):
        if False:
            return 10
        self.clear()

    def _updateClearButton(self):
        if False:
            return 10
        self._clearButton.setVisible(self._hasClearableContent())

    def _hasClearableContent(self):
        if False:
            return 10
        return bool(self.text())

    def resizeEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        if self._is_clearable:
            frame_width = self.style().pixelMetric(QStyle.PM_DefaultFrameWidth)
            rect = self.rect()
            right_hint = self._clearButton.sizeHint()
            right_x = rect.right() - frame_width - right_hint.width()
            right_y = (rect.bottom() - right_hint.height()) // 2
            self._clearButton.move(right_x, right_y)

    def _textChanged(self, text):
        if False:
            for i in range(10):
                print('nop')
        if self._is_clearable:
            self._updateClearButton()

class SearchEdit(ClearableEdit):

    def __init__(self, parent=None, immediate=False):
        if False:
            i = 10
            return i + 15
        ClearableEdit.__init__(self, parent, is_clearable=True)
        self.inactiveText = tr('Search...')
        self.immediate = immediate
        self.returnPressed.connect(self._returnPressed)

    def _clearSearch(self):
        if False:
            return 10
        ClearableEdit._clearSearch(self)
        self.searchChanged.emit()

    def _textChanged(self, text):
        if False:
            return 10
        ClearableEdit._textChanged(self, text)
        if self.immediate:
            self.searchChanged.emit()

    def keyPressEvent(self, event):
        if False:
            return 10
        key = event.key()
        if key == Qt.Key_Escape:
            self._clearSearch()
        else:
            ClearableEdit.keyPressEvent(self, event)

    def paintEvent(self, event):
        if False:
            return 10
        ClearableEdit.paintEvent(self, event)
        if not bool(self.text()) and self.inactiveText and (not self.hasFocus()):
            panel = QStyleOptionFrame()
            self.initStyleOption(panel)
            text_rect = self.style().subElementRect(QStyle.SE_LineEditContents, panel, self)
            left_margin = 2
            right_margin = self._clearButton.iconSize().width()
            text_rect.adjust(left_margin, 0, -right_margin, 0)
            painter = QPainter(self)
            disabled_color = self.palette().brush(QPalette.Disabled, QPalette.Text).color()
            painter.setPen(disabled_color)
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, self.inactiveText)

    def _returnPressed(self):
        if False:
            i = 10
            return i + 15
        if not self.immediate:
            self.searchChanged.emit()
    searchChanged = pyqtSignal()