import time
from datetime import datetime
from typing import Optional, Any
from PyQt5.QtCore import Qt, QDateTime, pyqtSignal
from PyQt5.QtGui import QPalette, QPainter
from PyQt5.QtWidgets import QWidget, QLineEdit, QStyle, QStyleOptionFrame, QComboBox, QHBoxLayout, QDateTimeEdit
from electrum.i18n import _
from electrum.bitcoin import NLOCKTIME_MIN, NLOCKTIME_MAX, NLOCKTIME_BLOCKHEIGHT_MAX
from .util import char_width_in_lineedit, ColorScheme

class LockTimeEdit(QWidget):
    valueEdited = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        QWidget.__init__(self, parent)
        hbox = QHBoxLayout()
        self.setLayout(hbox)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        self.locktime_raw_e = LockTimeRawEdit(self)
        self.locktime_height_e = LockTimeHeightEdit(self)
        self.locktime_date_e = LockTimeDateEdit(self)
        self.editors = [self.locktime_raw_e, self.locktime_height_e, self.locktime_date_e]
        self.combo = QComboBox()
        options = [_('Raw'), _('Block height'), _('Date')]
        option_index_to_editor_map = {0: self.locktime_raw_e, 1: self.locktime_height_e, 2: self.locktime_date_e}
        default_index = 1
        self.combo.addItems(options)

        def on_current_index_changed(i):
            if False:
                i = 10
                return i + 15
            for w in self.editors:
                w.setVisible(False)
                w.setEnabled(False)
            prev_locktime = self.editor.get_locktime()
            self.editor = option_index_to_editor_map[i]
            if self.editor.is_acceptable_locktime(prev_locktime):
                self.editor.set_locktime(prev_locktime)
            self.editor.setVisible(True)
            self.editor.setEnabled(True)
        self.editor = option_index_to_editor_map[default_index]
        self.combo.currentIndexChanged.connect(on_current_index_changed)
        self.combo.setCurrentIndex(default_index)
        on_current_index_changed(default_index)
        hbox.addWidget(self.combo)
        for w in self.editors:
            hbox.addWidget(w)
        hbox.addStretch(1)
        self.locktime_height_e.textEdited.connect(self.valueEdited.emit)
        self.locktime_raw_e.textEdited.connect(self.valueEdited.emit)
        self.locktime_date_e.dateTimeChanged.connect(self.valueEdited.emit)
        self.combo.currentIndexChanged.connect(self.valueEdited.emit)

    def get_locktime(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        return self.editor.get_locktime()

    def set_locktime(self, x: Any) -> None:
        if False:
            return 10
        self.editor.set_locktime(x)

class _LockTimeEditor:
    min_allowed_value = NLOCKTIME_MIN
    max_allowed_value = NLOCKTIME_MAX

    def get_locktime(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def set_locktime(self, x: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @classmethod
    def is_acceptable_locktime(cls, x: Any) -> bool:
        if False:
            while True:
                i = 10
        if not x:
            return True
        try:
            x = int(x)
        except Exception:
            return False
        return cls.min_allowed_value <= x <= cls.max_allowed_value

class LockTimeRawEdit(QLineEdit, _LockTimeEditor):

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        QLineEdit.__init__(self, parent)
        self.setFixedWidth(14 * char_width_in_lineedit())
        self.textChanged.connect(self.numbify)

    def numbify(self):
        if False:
            i = 10
            return i + 15
        text = self.text().strip()
        chars = '0123456789'
        pos = self.cursorPosition()
        pos = len(''.join([i for i in text[:pos] if i in chars]))
        s = ''.join([i for i in text if i in chars])
        self.set_locktime(s)
        self.setModified(self.hasFocus())
        self.setCursorPosition(pos)

    def get_locktime(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        try:
            return int(str(self.text()))
        except Exception:
            return None

    def set_locktime(self, x: Any) -> None:
        if False:
            print('Hello World!')
        try:
            x = int(x)
        except Exception:
            self.setText('')
            return
        x = max(x, self.min_allowed_value)
        x = min(x, self.max_allowed_value)
        self.setText(str(x))

class LockTimeHeightEdit(LockTimeRawEdit):
    max_allowed_value = NLOCKTIME_BLOCKHEIGHT_MAX

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        LockTimeRawEdit.__init__(self, parent)
        self.setFixedWidth(20 * char_width_in_lineedit())

    def paintEvent(self, event):
        if False:
            return 10
        super().paintEvent(event)
        panel = QStyleOptionFrame()
        self.initStyleOption(panel)
        textRect = self.style().subElementRect(QStyle.SE_LineEditContents, panel, self)
        textRect.adjust(2, 0, -10, 0)
        painter = QPainter(self)
        painter.setPen(ColorScheme.GRAY.as_color())
        painter.drawText(textRect, int(Qt.AlignRight | Qt.AlignVCenter), 'height')

def get_max_allowed_timestamp() -> int:
    if False:
        return 10
    ts = NLOCKTIME_MAX
    try:
        datetime.fromtimestamp(ts)
    except (OSError, OverflowError):
        ts = 2 ** 31 - 1
        datetime.fromtimestamp(ts)
    return ts

class LockTimeDateEdit(QDateTimeEdit, _LockTimeEditor):
    min_allowed_value = NLOCKTIME_BLOCKHEIGHT_MAX + 1
    max_allowed_value = get_max_allowed_timestamp()

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        QDateTimeEdit.__init__(self, parent)
        self.setMinimumDateTime(datetime.fromtimestamp(self.min_allowed_value))
        self.setMaximumDateTime(datetime.fromtimestamp(self.max_allowed_value))
        self.setDateTime(QDateTime.currentDateTime())

    def get_locktime(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        dt = self.dateTime().toPyDateTime()
        locktime = int(time.mktime(dt.timetuple()))
        return locktime

    def set_locktime(self, x: Any) -> None:
        if False:
            while True:
                i = 10
        if not self.is_acceptable_locktime(x):
            self.setDateTime(QDateTime.currentDateTime())
            return
        try:
            x = int(x)
        except Exception:
            self.setDateTime(QDateTime.currentDateTime())
            return
        dt = datetime.fromtimestamp(x)
        self.setDateTime(dt)