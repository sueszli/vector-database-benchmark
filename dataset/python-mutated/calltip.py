"""
Calltip widget used only to show signatures.

Adapted from IPython/frontend/qt/console/call_tip_widget.py of the
`IPython Project <https://github.com/ipython/ipython>`_.
Now located at qtconsole/call_tip_widget.py as part of the
`Jupyter QtConsole Project <https://github.com/jupyter/qtconsole>`_.
"""
from unicodedata import category
import os
import sys
from qtpy.QtCore import QBasicTimer, QCoreApplication, QEvent, Qt, QTimer, Signal
from qtpy.QtGui import QCursor, QPalette
from qtpy.QtWidgets import QApplication, QFrame, QLabel, QTextEdit, QPlainTextEdit, QStyle, QStyleOptionFrame, QStylePainter, QToolTip
from spyder.py3compat import to_text_string

class ToolTipWidget(QLabel):
    """
    Shows tooltips that can be styled with the different themes.
    """
    sig_completion_help_requested = Signal(str, str)
    sig_help_requested = Signal(str)

    def __init__(self, parent=None, as_tooltip=False):
        if False:
            while True:
                i = 10
        '\n        Shows tooltips that can be styled with the different themes.\n        '
        super(ToolTipWidget, self).__init__(None, Qt.ToolTip)
        self.completion_doc = None
        self._url = ''
        self.app = QCoreApplication.instance()
        self.as_tooltip = as_tooltip
        self.tip = None
        self._timer_hide = QTimer()
        self._text_edit = parent
        if sys.platform == 'darwin':
            self.setWindowFlags(Qt.SplashScreen)
        else:
            self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self._timer_hide.setInterval(500)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.setOpenExternalLinks(False)
        self.setForegroundRole(QPalette.ToolTipText)
        self.setBackgroundRole(QPalette.ToolTipBase)
        self.setPalette(QToolTip.palette())
        self.setAlignment(Qt.AlignLeft)
        self.setIndent(1)
        self.setFrameStyle(QFrame.NoFrame)
        style = self.style()
        delta_margin = style.pixelMetric(QStyle.PM_ToolTipLabelFrameWidth, None, self)
        self.setMargin(1 + delta_margin)
        self.linkHovered.connect(self._update_hover_html_link_style)
        self._timer_hide.timeout.connect(self.hide)
        QApplication.instance().applicationStateChanged.connect(self._should_hide)

    def paintEvent(self, event):
        if False:
            while True:
                i = 10
        'Reimplemented to paint the background panel.'
        painter = QStylePainter(self)
        option = QStyleOptionFrame()
        option.initFrom(self)
        painter.drawPrimitive(QStyle.PE_PanelTipLabel, option)
        painter.end()
        super(ToolTipWidget, self).paintEvent(event)

    def _update_hover_html_link_style(self, url):
        if False:
            for i in range(10):
                print('nop')
        'Update style of labels that include rich text and html links.'
        link = 'text-decoration:none;'
        link_hovered = 'text-decoration:underline;'
        self._url = url
        if url:
            QApplication.setOverrideCursor(QCursor(Qt.PointingHandCursor))
            (new_text, old_text) = (link_hovered, link)
        else:
            (new_text, old_text) = (link, link_hovered)
            QApplication.restoreOverrideCursor()
        text = self.text()
        new_text = text.replace(old_text, new_text)
        self.setText(new_text)

    def show_basic_tip(self, point, tip):
        if False:
            i = 10
            return i + 15
        'Show basic tip.'
        self.tip = tip
        self.setText(tip)
        self.resize(self.sizeHint())
        y = point.y() - self.height()
        self.move(point.x(), y)
        self.show()
        return True

    def show_tip(self, point, tip, cursor=None, completion_doc=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Attempts to show the specified tip at the current cursor location.\n        '
        if QApplication.instance().applicationState() != Qt.ApplicationActive:
            return
        self.tip = tip
        self.setText(tip)
        self.resize(self.sizeHint())
        self.completion_doc = completion_doc
        padding = 0
        text_edit = self._text_edit
        if cursor is None:
            cursor_rect = text_edit.cursorRect()
        else:
            cursor_rect = text_edit.cursorRect(cursor)
        screen_rect = self.app.desktop().screenGeometry(text_edit)
        point.setY(point.y() + padding)
        tip_height = self.size().height()
        tip_width = self.size().width()
        vertical = 'bottom'
        horizontal = 'Right'
        if point.y() + tip_height > screen_rect.height() + screen_rect.y():
            point_ = text_edit.mapToGlobal(cursor_rect.topRight())
            if point_.y() - tip_height < padding:
                if 2 * point.y() < screen_rect.height():
                    vertical = 'bottom'
                else:
                    vertical = 'top'
            else:
                vertical = 'top'
        if point.x() + tip_width > screen_rect.width() + screen_rect.x():
            point_ = text_edit.mapToGlobal(cursor_rect.topRight())
            if point_.x() - tip_width < padding:
                if 2 * point.x() < screen_rect.width():
                    horizontal = 'Right'
                else:
                    horizontal = 'Left'
            else:
                horizontal = 'Left'
        pos = getattr(cursor_rect, '%s%s' % (vertical, horizontal))
        adjusted_point = text_edit.mapToGlobal(pos())
        if vertical == 'top':
            if os.name == 'nt':
                padding = -7
            else:
                padding = -4
            point.setY(adjusted_point.y() - tip_height - padding)
        if horizontal == 'Left':
            point.setX(adjusted_point.x() - tip_width - padding)
        self.move(point)
        if not self.isVisible():
            self.show()
        return True

    def mousePressEvent(self, event):
        if False:
            print('Hello World!')
        '\n        Reimplemented to hide it when focus goes out of the main window.\n        '
        QApplication.restoreOverrideCursor()
        if self.completion_doc:
            name = self.completion_doc.get('name', '')
            signature = self.completion_doc.get('signature', '')
            self.sig_completion_help_requested.emit(name, signature)
        else:
            self.sig_help_requested.emit(self._url)
        super(ToolTipWidget, self).mousePressEvent(event)
        self._hide()

    def focusOutEvent(self, event):
        if False:
            return 10
        '\n        Reimplemented to hide it when focus goes out of the main window.\n        '
        self.hide()

    def leaveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Override Qt method to hide the tooltip on leave.'
        super(ToolTipWidget, self).leaveEvent(event)
        self.hide()

    def _hide(self):
        if False:
            i = 10
            return i + 15
        'Hide method helper.'
        QApplication.restoreOverrideCursor()
        self._timer_hide.start()

    def hide(self):
        if False:
            while True:
                i = 10
        'Override Qt method to add timer and restore cursor.'
        super(ToolTipWidget, self).hide()
        self._timer_hide.stop()

    def _should_hide(self, state):
        if False:
            i = 10
            return i + 15
        '\n        This widget should hide itself if the application is not active.\n        '
        if state != Qt.ApplicationActive:
            self.hide()

class CallTipWidget(QLabel):
    """ Shows call tips by parsing the current text of Q[Plain]TextEdit.
    """

    def __init__(self, text_edit, hide_timer_on=False, as_tooltip=False):
        if False:
            i = 10
            return i + 15
        ' Create a call tip manager that is attached to the specified Qt\n            text edit widget.\n        '
        assert isinstance(text_edit, (QTextEdit, QPlainTextEdit))
        super(CallTipWidget, self).__init__(None, Qt.ToolTip)
        self.app = QCoreApplication.instance()
        self.as_tooltip = as_tooltip
        self.hide_timer_on = hide_timer_on
        self.tip = None
        self._hide_timer = QBasicTimer()
        self._text_edit = text_edit
        self._start_position = -1
        if sys.platform == 'darwin':
            self.setWindowFlags(Qt.SplashScreen)
        else:
            self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setFont(text_edit.document().defaultFont())
        self.setForegroundRole(QPalette.ToolTipText)
        self.setBackgroundRole(QPalette.ToolTipBase)
        self.setPalette(QToolTip.palette())
        self.setAlignment(Qt.AlignLeft)
        self.setIndent(1)
        self.setFrameStyle(QFrame.NoFrame)
        self.setMargin(1 + self.style().pixelMetric(QStyle.PM_ToolTipLabelFrameWidth, None, self))
        QApplication.instance().applicationStateChanged.connect(self._should_hide)

    def eventFilter(self, obj, event):
        if False:
            print('Hello World!')
        ' Reimplemented to hide on certain key presses and on text edit focus\n            changes.\n        '
        if obj == self._text_edit:
            etype = event.type()
            if etype == QEvent.KeyPress:
                key = event.key()
                cursor = self._text_edit.textCursor()
                prev_char = self._text_edit.get_character(cursor.position(), offset=-1)
                if key in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Down, Qt.Key_Up):
                    self.hide()
                elif key == Qt.Key_Escape:
                    self.hide()
                    return True
                elif prev_char == ')':
                    self.hide()
            elif etype == QEvent.FocusOut:
                self.hide()
            elif etype == QEvent.Enter:
                if self._hide_timer.isActive() and self.app.topLevelAt(QCursor.pos()) == self:
                    self._hide_timer.stop()
            elif etype == QEvent.Leave:
                self._leave_event_hide()
            elif etype == QEvent.WindowBlocked:
                self.hide()
        return super(CallTipWidget, self).eventFilter(obj, event)

    def timerEvent(self, event):
        if False:
            return 10
        ' Reimplemented to hide the widget when the hide timer fires.\n        '
        if event.timerId() == self._hide_timer.timerId():
            self._hide_timer.stop()
            self.hide()

    def enterEvent(self, event):
        if False:
            print('Hello World!')
        ' Reimplemented to cancel the hide timer.\n        '
        super(CallTipWidget, self).enterEvent(event)
        if self.as_tooltip:
            self.hide()
        if self._hide_timer.isActive() and self.app.topLevelAt(QCursor.pos()) == self:
            self._hide_timer.stop()

    def hideEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        ' Reimplemented to disconnect signal handlers and event filter.\n        '
        super(CallTipWidget, self).hideEvent(event)
        try:
            self._text_edit.cursorPositionChanged.disconnect(self._cursor_position_changed)
        except (TypeError, RuntimeError):
            pass
        else:
            self._text_edit.removeEventFilter(self)

    def leaveEvent(self, event):
        if False:
            i = 10
            return i + 15
        ' Reimplemented to start the hide timer.\n        '
        super(CallTipWidget, self).leaveEvent(event)
        self._leave_event_hide()

    def mousePressEvent(self, event):
        if False:
            i = 10
            return i + 15
        super(CallTipWidget, self).mousePressEvent(event)
        self.hide()

    def paintEvent(self, event):
        if False:
            return 10
        ' Reimplemented to paint the background panel.\n        '
        painter = QStylePainter(self)
        option = QStyleOptionFrame()
        option.initFrom(self)
        painter.drawPrimitive(QStyle.PE_PanelTipLabel, option)
        painter.end()
        super(CallTipWidget, self).paintEvent(event)

    def setFont(self, font):
        if False:
            print('Hello World!')
        ' Reimplemented to allow use of this method as a slot.\n        '
        super(CallTipWidget, self).setFont(font)

    def showEvent(self, event):
        if False:
            while True:
                i = 10
        ' Reimplemented to connect signal handlers and event filter.\n        '
        super(CallTipWidget, self).showEvent(event)
        self._text_edit.cursorPositionChanged.connect(self._cursor_position_changed)
        self._text_edit.installEventFilter(self)

    def focusOutEvent(self, event):
        if False:
            while True:
                i = 10
        ' Reimplemented to hide it when focus goes out of the main\n            window.\n        '
        self.hide()

    def show_tip(self, point, tip, wrapped_tiplines):
        if False:
            i = 10
            return i + 15
        ' Attempts to show the specified tip at the current cursor location.\n        '
        if QApplication.instance().applicationState() != Qt.ApplicationActive:
            return
        if self.isVisible():
            if self.tip == tip:
                return True
            else:
                self.hide()
        text_edit = self._text_edit
        cursor = text_edit.textCursor()
        search_pos = cursor.position() - 1
        (self._start_position, _) = self._find_parenthesis(search_pos, forward=False)
        if self._start_position == -1:
            return False
        if self.hide_timer_on:
            self._hide_timer.stop()
            if len(wrapped_tiplines) == 1:
                args = wrapped_tiplines[0].split('(')[1]
                nargs = len(args.split(','))
                if nargs == 1:
                    hide_time = 1400
                elif nargs == 2:
                    hide_time = 1600
                else:
                    hide_time = 1800
            elif len(wrapped_tiplines) == 2:
                args1 = wrapped_tiplines[1].strip()
                nargs1 = len(args1.split(','))
                if nargs1 == 1:
                    hide_time = 2500
                else:
                    hide_time = 2800
            else:
                hide_time = 3500
            self._hide_timer.start(hide_time, self)
        self.tip = tip
        self.setText(tip)
        self.resize(self.sizeHint())
        padding = 0
        cursor_rect = text_edit.cursorRect(cursor)
        screen_rect = self.app.desktop().screenGeometry(text_edit)
        point.setY(point.y() + padding)
        tip_height = self.size().height()
        tip_width = self.size().width()
        vertical = 'bottom'
        horizontal = 'Right'
        if point.y() + tip_height > screen_rect.height() + screen_rect.y():
            point_ = text_edit.mapToGlobal(cursor_rect.topRight())
            if point_.y() - tip_height < padding:
                if 2 * point.y() < screen_rect.height():
                    vertical = 'bottom'
                else:
                    vertical = 'top'
            else:
                vertical = 'top'
        if point.x() + tip_width > screen_rect.width() + screen_rect.x():
            point_ = text_edit.mapToGlobal(cursor_rect.topRight())
            if point_.x() - tip_width < padding:
                if 2 * point.x() < screen_rect.width():
                    horizontal = 'Right'
                else:
                    horizontal = 'Left'
            else:
                horizontal = 'Left'
        pos = getattr(cursor_rect, '%s%s' % (vertical, horizontal))
        adjusted_point = text_edit.mapToGlobal(pos())
        if vertical == 'top':
            if os.name == 'nt':
                padding = -7
            else:
                padding = -4
            point.setY(adjusted_point.y() - tip_height - padding)
        if horizontal == 'Left':
            point.setX(adjusted_point.x() - tip_width - padding)
        self.move(point)
        self.show()
        return True

    def _find_parenthesis(self, position, forward=True):
        if False:
            return 10
        " If 'forward' is True (resp. False), proceed forwards\n            (resp. backwards) through the line that contains 'position' until an\n            unmatched closing (resp. opening) parenthesis is found. Returns a\n            tuple containing the position of this parenthesis (or -1 if it is\n            not found) and the number commas (at depth 0) found along the way.\n        "
        commas = depth = 0
        document = self._text_edit.document()
        char = to_text_string(document.characterAt(position))
        while category(char) != 'Cc' and position > 0:
            if char == ',' and depth == 0:
                commas += 1
            elif char == ')':
                if forward and depth == 0:
                    break
                depth += 1
            elif char == '(':
                if not forward and depth == 0:
                    break
                depth -= 1
            position += 1 if forward else -1
            char = to_text_string(document.characterAt(position))
        else:
            position = -1
        return (position, commas)

    def _leave_event_hide(self):
        if False:
            while True:
                i = 10
        ' Hides the tooltip after some time has passed (assuming the cursor is\n            not over the tooltip).\n        '
        if self.hide_timer_on and (not self._hide_timer.isActive()) and (self.app.topLevelAt(QCursor.pos()) != self):
            self._hide_timer.start(800, self)

    def _cursor_position_changed(self):
        if False:
            print('Hello World!')
        ' Updates the tip based on user cursor movement.\n        '
        cursor = self._text_edit.textCursor()
        position = cursor.position()
        document = self._text_edit.document()
        char = to_text_string(document.characterAt(position - 1))
        if position <= self._start_position:
            self.hide()
        elif char == ')':
            (pos, _) = self._find_parenthesis(position - 1, forward=False)
            if pos == -1:
                self.hide()

    def _should_hide(self, state):
        if False:
            for i in range(10):
                print('nop')
        '\n        This widget should hide itself if the application is not active.\n        '
        if state != Qt.ApplicationActive:
            self.hide()