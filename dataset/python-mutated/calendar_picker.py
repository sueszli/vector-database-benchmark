from typing import Union
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QDate, QPoint, pyqtProperty
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication
from ...common.style_sheet import FluentStyleSheet
from ...common.icon import FluentIcon as FIF
from .calendar_view import CalendarView

class CalendarPicker(QPushButton):
    """ Calendar picker """
    dateChanged = pyqtSignal(QDate)

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)
        self._date = QDate()
        self._dateFormat = Qt.SystemLocaleDate
        self.setText(self.tr('Pick a date'))
        FluentStyleSheet.CALENDAR_PICKER.apply(self)
        self.clicked.connect(self._showCalendarView)

    def getDate(self):
        if False:
            i = 10
            return i + 15
        return self._date

    def setDate(self, date: QDate):
        if False:
            while True:
                i = 10
        ' set the selected date '
        self._onDateChanged(date)

    def getDateFormat(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dateFormat

    def setDateFormat(self, format: Union[Qt.DateFormat, str]):
        if False:
            while True:
                i = 10
        self._dateFormat = format
        if self.date.isValid():
            self.setText(self.date.toString(self.dateFormat))

    def _showCalendarView(self):
        if False:
            i = 10
            return i + 15
        view = CalendarView(self.window())
        view.dateChanged.connect(self._onDateChanged)
        if self.date.isValid():
            view.setDate(self.date)
        x = int(self.width() / 2 - view.sizeHint().width() / 2)
        y = self.height()
        view.exec(self.mapToGlobal(QPoint(x, y)))

    def _onDateChanged(self, date: QDate):
        if False:
            return 10
        self._date = QDate(date)
        self.setText(date.toString(self.dateFormat))
        self.setProperty('hasDate', True)
        self.setStyle(QApplication.style())
        self.update()
        self.dateChanged.emit(date)

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        if not self.property('hasDate'):
            painter.setOpacity(0.6)
        w = 12
        rect = QRectF(self.width() - 23, self.height() / 2 - w / 2, w, w)
        FIF.CALENDAR.render(painter, rect)
    date = pyqtProperty(QDate, getDate, setDate)
    dateFormat = pyqtProperty(Qt.DateFormat, getDateFormat, setDateFormat)