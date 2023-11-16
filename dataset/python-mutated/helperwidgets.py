"""
Helper widgets.
"""
import re
import qtawesome as qta
import qstylizer.style
from qtpy import PYQT5
from qtpy.QtCore import QEvent, QPoint, QRegExp, QSize, QSortFilterProxyModel, Qt, Signal
from qtpy.QtGui import QAbstractTextDocumentLayout, QColor, QFontMetrics, QImage, QPainter, QRegExpValidator, QTextDocument, QPixmap
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtWidgets import QAction, QApplication, QCheckBox, QLineEdit, QMessageBox, QSpacerItem, QStyle, QStyledItemDelegate, QStyleOptionFrame, QStyleOptionViewItem, QTableView, QToolButton, QToolTip, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QFrame, QComboBox
from spyder.api.config.fonts import SpyderFontType, SpyderFontsMixin
from spyder.api.config.mixins import SpyderConfigurationAccessor
from spyder.config.base import _
from spyder.utils.icon_manager import ima
from spyder.utils.stringmatching import get_search_regex
from spyder.utils.palette import QStylePalette, SpyderPalette
from spyder.utils.image_path_manager import get_image_path
from spyder.utils.stylesheet import AppStyle, DialogStyle
from spyder.utils.qthelpers import create_waitspinner
VALID_ACCENT_CHARS = 'ÁÉÍOÚáéíúóàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛäëïöüÄËÏÖÜñÑ'
VALID_FINDER_CHARS = '[A-Za-z\\s{0}]'.format(VALID_ACCENT_CHARS)

class HelperToolButton(QToolButton):
    """Subclasses QToolButton, to provide a simple tooltip on mousedown.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        QToolButton.__init__(self)
        self.setIcon(ima.get_std_icon('MessageBoxInformation'))
        style = '\n            QToolButton {\n              padding:0px;\n              border-radius: 2px;\n            }\n            '
        self.setStyleSheet(style)

    def setToolTip(self, text):
        if False:
            print('Hello World!')
        self._tip_text = text

    def toolTip(self):
        if False:
            print('Hello World!')
        return self._tip_text

    def mousePressEvent(self, event):
        if False:
            print('Hello World!')
        QToolTip.hideText()

    def mouseReleaseEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        QToolTip.showText(self.mapToGlobal(QPoint(0, self.height())), self._tip_text)

class MessageCheckBox(QMessageBox):
    """
    A QMessageBox derived widget that includes a QCheckBox aligned to the right
    under the message and on top of the buttons.
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(MessageCheckBox, self).__init__(*args, **kwargs)
        self.setWindowModality(Qt.NonModal)
        self._checkbox = QCheckBox(self)
        size = 9
        check_layout = QVBoxLayout()
        check_layout.addItem(QSpacerItem(size, size))
        check_layout.addWidget(self._checkbox, 0, Qt.AlignRight)
        check_layout.addItem(QSpacerItem(size, size))
        layout = self.layout()
        if PYQT5:
            layout.addLayout(check_layout, 1, 2)
        else:
            layout.addLayout(check_layout, 1, 1)

    def is_checked(self):
        if False:
            while True:
                i = 10
        return self._checkbox.isChecked()

    def set_checked(self, value):
        if False:
            for i in range(10):
                print('nop')
        return self._checkbox.setChecked(value)

    def set_check_visible(self, value):
        if False:
            while True:
                i = 10
        self._checkbox.setVisible(value)

    def is_check_visible(self):
        if False:
            print('Hello World!')
        self._checkbox.isVisible()

    def checkbox_text(self):
        if False:
            i = 10
            return i + 15
        self._checkbox.text()

    def set_checkbox_text(self, text):
        if False:
            while True:
                i = 10
        self._checkbox.setText(text)

class HTMLDelegate(QStyledItemDelegate):
    """
    With this delegate, a QListWidgetItem or a QTableItem can render HTML.

    Taken from https://stackoverflow.com/a/5443112/2399799
    """

    def __init__(self, parent, margin=0, wrap_text=False, align_vcenter=False):
        if False:
            i = 10
            return i + 15
        super(HTMLDelegate, self).__init__(parent)
        self._margin = margin
        self._wrap_text = wrap_text
        self._hovered_row = -1
        self._align_vcenter = align_vcenter

    def _prepare_text_document(self, option, index):
        if False:
            while True:
                i = 10
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        doc = QTextDocument()
        doc.setDocumentMargin(self._margin)
        doc.setHtml(options.text)
        if self._wrap_text:
            doc.setTextWidth(option.rect.width() - 25)
        return (options, doc)

    def on_hover_index_changed(self, index):
        if False:
            i = 10
            return i + 15
        "\n        This can be used by a widget that inherits from HoverRowsTableView to\n        connect its sig_hover_index_changed signal to this method to paint an\n        entire row when it's hovered.\n        "
        self._hovered_row = index.row()

    def paint(self, painter, option, index):
        if False:
            print('Hello World!')
        (options, doc) = self._prepare_text_document(option, index)
        style = QApplication.style() if options.widget is None else options.widget.style()
        options.text = ''
        if index.row() == self._hovered_row:
            painter.fillRect(options.rect, QColor(QStylePalette.COLOR_BACKGROUND_3))
        style.drawControl(QStyle.CE_ItemViewItem, options, painter, options.widget)
        ctx = QAbstractTextDocumentLayout.PaintContext()
        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options, None)
        painter.save()
        if hasattr(options.widget, 'files_list'):
            if options.widget.files_list:
                painter.translate(textRect.topLeft() + QPoint(4, 4))
            else:
                painter.translate(textRect.topLeft() + QPoint(2, 4))
        elif not self._align_vcenter:
            painter.translate(textRect.topLeft() + QPoint(0, -3))
        if self._align_vcenter:
            doc.setTextWidth(option.rect.width())
            offset_y = (option.rect.height() - doc.size().height()) / 2
            painter.translate(options.rect.x(), options.rect.y() + offset_y)
            doc.drawContents(painter)
        docLayout = doc.documentLayout()
        if type(docLayout) is QAbstractTextDocumentLayout:
            docLayout.draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        if False:
            return 10
        (__, doc) = self._prepare_text_document(option, index)
        return QSize(round(doc.idealWidth()), round(doc.size().height() - 2))

class ItemDelegate(QStyledItemDelegate):

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        QStyledItemDelegate.__init__(self, parent)

    def paint(self, painter, option, index):
        if False:
            while True:
                i = 10
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        style = QApplication.style() if options.widget is None else options.widget.style()
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setHtml(options.text)
        options.text = ''
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)
        ctx = QAbstractTextDocumentLayout.PaintContext()
        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options, None)
        painter.save()
        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        if False:
            return 10
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        doc = QTextDocument()
        doc.setHtml(options.text)
        doc.setTextWidth(options.rect.width())
        return QSize(int(doc.idealWidth()), int(doc.size().height()))

class IconLineEdit(QLineEdit):
    """
    Custom QLineEdit that includes an icon representing a validation for its
    text and can also elide it.
    """

    def __init__(self, parent, elide_text=False, ellipsis_place=Qt.ElideLeft):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.elide_text = elide_text
        self.ellipsis_place = ellipsis_place
        self._status = True
        self._status_set = True
        self._focus_in = False
        self._valid_icon = ima.icon('todo')
        self._invalid_icon = ima.icon('warning')
        self._set_icon = ima.icon('todo_list')
        self._refresh()
        self._paint_count = 0
        self._icon_visible = False

    def _refresh(self):
        if False:
            while True:
                i = 10
        '\n        This makes space for the right validation icons after focus is given to\n        the widget.\n        '
        padding = self.height()
        if self.elide_text and (not self._focus_in):
            padding = 0
        css = qstylizer.style.StyleSheet()
        css.QLineEdit.setValues(border='none', paddingRight=f'{padding}px')
        self.setStyleSheet(css.toString())
        self.update()

    def hide_status_icon(self):
        if False:
            print('Hello World!')
        'Show the status icon.'
        self._icon_visible = False
        self.repaint()
        self.update()

    def show_status_icon(self):
        if False:
            print('Hello World!')
        'Hide the status icon.'
        self._icon_visible = True
        self.repaint()
        self.update()

    def update_status(self, value, value_set):
        if False:
            for i in range(10):
                print('nop')
        'Update the status and set_status to update the icons to display.'
        self._status = value
        self._status_set = value_set
        self.repaint()
        self.update()

    def paintEvent(self, event):
        if False:
            while True:
                i = 10
        '\n        Include a validation icon to the left of the line edit and elide text\n        if requested.\n        '
        if self.elide_text and (not self._focus_in):
            painter = QPainter(self)
            option = QStyleOptionFrame()
            self.initStyleOption(option)
            text_rect = self.style().subElementRect(QStyle.SE_LineEditContents, option, self)
            text_rect.adjust(0, 0, -2, 0)
            fm = QFontMetrics(self.font())
            text = fm.elidedText(self.text(), self.ellipsis_place, text_rect.width())
            painter.setPen(QColor(QStylePalette.COLOR_TEXT_1))
            painter.drawText(text_rect, int(Qt.AlignLeft | Qt.AlignVCenter), text)
            return
        super().paintEvent(event)
        painter = QPainter(self)
        rect = self.geometry()
        space = int(rect.height() / 6)
        h = rect.height() - space
        w = rect.width() - h
        if self._icon_visible:
            if self._status and self._status_set:
                pixmap = self._set_icon.pixmap(h, h)
            elif self._status:
                pixmap = self._valid_icon.pixmap(h, h)
            else:
                pixmap = self._invalid_icon.pixmap(h, h)
            painter.drawPixmap(w, 2, pixmap)
        if self._paint_count < 5:
            self._paint_count += 1
            self._refresh()

    def focusInEvent(self, event):
        if False:
            return 10
        'Reimplemented to know when this widget has received focus.'
        self._focus_in = True
        self._refresh()
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Reimplemented to know when this widget has lost focus.'
        self._focus_in = False
        self._refresh()
        super().focusOutEvent(event)

class ClearLineEdit(QLineEdit):
    """QLineEdit with a clear button."""

    def __init__(self, parent, reposition_button=False):
        if False:
            return 10
        super().__init__(parent)
        self.clear_action = QAction(self)
        self.clear_action.setIcon(ima.icon('clear_text'))
        self.clear_action.setToolTip(_('Clear text'))
        self.clear_action.triggered.connect(self.clear)
        self.addAction(self.clear_action, QLineEdit.TrailingPosition)
        self.clear_button = self.findChildren(QToolButton)[0]
        self.clear_action.setVisible(False)
        self.textChanged.connect(self._on_text_changed)
        if reposition_button:
            self.installEventFilter(self)

    def _on_text_changed(self, text):
        if False:
            i = 10
            return i + 15
        'Actions to take when text has changed on the line edit widget.'
        if text:
            self.clear_action.setVisible(True)
        else:
            self.clear_action.setVisible(False)

    def eventFilter(self, widget, event):
        if False:
            i = 10
            return i + 15
        '\n        Event filter for this widget used to reduce the space between\n        clear_button and the right border of the line edit.\n        '
        if event.type() == QEvent.Paint:
            self.clear_button.move(self.width() - 22, self.clear_button.y())
        return super().eventFilter(widget, event)

class FinderLineEdit(ClearLineEdit):
    sig_hide_requested = Signal()
    sig_find_requested = Signal()

    def __init__(self, parent, regex_base=None, key_filter_dict=None):
        if False:
            return 10
        super().__init__(parent)
        self.key_filter_dict = key_filter_dict
        self._combobox = QComboBox(self)
        self._is_shown = False
        if regex_base is not None:
            regex = QRegExp(regex_base + '{100}')
            self.setValidator(QRegExpValidator(regex))

    def keyPressEvent(self, event):
        if False:
            print('Hello World!')
        'Qt and FilterLineEdit Override.'
        key = event.key()
        if self.key_filter_dict is not None and key in self.key_filter_dict:
            self.key_filter_dict[key]()
        elif key in [Qt.Key_Escape]:
            self.sig_hide_requested.emit()
        elif key in [Qt.Key_Enter, Qt.Key_Return]:
            self.sig_find_requested.emit()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Adjustments when the widget is shown.'
        if not self._is_shown:
            height = self._combobox.size().height()
            self._combobox.hide()
            self.setMinimumWidth(AppStyle.FindMinWidth)
            self.setMinimumHeight(height)
            self.setMaximumHeight(height)
            self._is_shown = True
        super().showEvent(event)

class FinderWidget(QWidget):
    sig_find_text = Signal(str)
    sig_hide_finder_requested = Signal()

    def __init__(self, parent, regex_base=None, key_filter_dict=None, find_on_change=False):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.text_finder = FinderLineEdit(self, regex_base=regex_base, key_filter_dict=key_filter_dict)
        self.text_finder.sig_find_requested.connect(self.do_find)
        if find_on_change:
            self.text_finder.textChanged.connect(self.do_find)
        self.text_finder.sig_hide_requested.connect(self.sig_hide_finder_requested)
        self.finder_close_button = QToolButton(self)
        self.finder_close_button.setIcon(ima.icon('DialogCloseButton'))
        self.finder_close_button.clicked.connect(self.sig_hide_finder_requested)
        finder_layout = QHBoxLayout()
        finder_layout.addWidget(self.finder_close_button)
        finder_layout.addWidget(self.text_finder)
        finder_layout.addStretch()
        finder_layout.setContentsMargins(2 * AppStyle.MarginSize, AppStyle.MarginSize, 2 * AppStyle.MarginSize, 0)
        self.setLayout(finder_layout)
        self.setVisible(False)

    def do_find(self):
        if False:
            i = 10
            return i + 15
        'Send text.'
        text = self.text_finder.text()
        if not text:
            text = ''
        self.sig_find_text.emit(text)

    def set_visible(self, visible):
        if False:
            while True:
                i = 10
        'Set visibility of widget.'
        self.setVisible(visible)
        if visible:
            self.text_finder.setFocus()
            self.do_find()
        else:
            self.sig_find_text.emit('')

class CustomSortFilterProxy(QSortFilterProxyModel):
    """Custom column filter based on regex."""

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super(CustomSortFilterProxy, self).__init__(parent)
        self._parent = parent
        self.pattern = re.compile('')

    def set_filter(self, text):
        if False:
            return 10
        'Set regular expression for filter.'
        self.pattern = get_search_regex(text)
        if self.pattern and text:
            self._parent.setSortingEnabled(False)
        else:
            self._parent.setSortingEnabled(True)
        self.invalidateFilter()

    def filterAcceptsRow(self, row_num, parent):
        if False:
            i = 10
            return i + 15
        'Qt override.\n\n        Reimplemented from base class to allow the use of custom filtering.\n        '
        model = self.sourceModel()
        name = model.row(row_num).name
        r = re.search(self.pattern, name)
        if r is None:
            return False
        else:
            return True

class PaneEmptyWidget(QFrame, SpyderConfigurationAccessor, SpyderFontsMixin):
    """Widget to show a pane/plugin functionality description."""

    def __init__(self, parent, icon_filename, text=None, description=None, top_stretch: int=1, middle_stretch: int=1, bottom_stretch: int=0, spinner: bool=False):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        interface_font_size = self.get_font(SpyderFontType.Interface).pointSize()
        image_label = QLabel(self)
        image_label.setPixmap(self.get_icon(icon_filename))
        image_label.setAlignment(Qt.AlignCenter)
        image_label_qss = qstylizer.style.StyleSheet()
        image_label_qss.QLabel.setValues(border='0px')
        image_label.setStyleSheet(image_label_qss.toString())
        if text is not None:
            text_label = QLabel(text, parent=self)
            text_label.setAlignment(Qt.AlignCenter)
            text_label.setWordWrap(True)
            text_label_qss = qstylizer.style.StyleSheet()
            text_label_qss.QLabel.setValues(fontSize=f'{interface_font_size + 5}pt', border='0px')
            text_label.setStyleSheet(text_label_qss.toString())
        if description is not None:
            description_label = QLabel(description, parent=self)
            description_label.setAlignment(Qt.AlignCenter)
            description_label.setWordWrap(True)
            description_label_qss = qstylizer.style.StyleSheet()
            description_label_qss.QLabel.setValues(fontSize=f'{interface_font_size}pt', backgroundColor=SpyderPalette.COLOR_OCCURRENCE_3, border='0px', padding='20px')
            description_label.setStyleSheet(description_label_qss.toString())
        pane_empty_layout = QVBoxLayout()
        pane_empty_layout.addStretch(top_stretch)
        pane_empty_layout.addWidget(image_label)
        if spinner is not False:
            spin_widget = qta.IconWidget()
            spin_icon = qta.icon('mdi.loading', color='white', animation=qta.Spin(spin_widget, interval=3))
            spin_widget.setIconSize(QSize(32, 32))
            spin_widget.setIcon(spin_icon)
            spin_widget.setStyleSheet(image_label_qss.toString())
            spin_widget.setAlignment(Qt.AlignCenter)
            pane_empty_layout.addWidget(spin_widget)
            pane_empty_layout.addItem(QSpacerItem(20, 20))
        if text is not None:
            pane_empty_layout.addWidget(text_label)
            pane_empty_layout.addStretch(middle_stretch)
        if description is not None:
            pane_empty_layout.addWidget(description_label)
        pane_empty_layout.addStretch(bottom_stretch)
        pane_empty_layout.setContentsMargins(20, 0, 20, 20)
        self.setLayout(pane_empty_layout)
        self.setFocusPolicy(Qt.StrongFocus)
        self._apply_stylesheet(False)

    def setup(self, *args, **kwargs):
        if False:
            return 10
        '\n        This method is needed when using this widget to show a "no connected\n        console" message in plugins that inherit from ShellConnectMainWidget.\n        '
        pass

    def get_icon(self, icon_filename):
        if False:
            return 10
        "\n        Get pane's icon as a QPixmap that it's scaled according to the factor\n        set by users in Preferences.\n        "
        image_path = get_image_path(icon_filename)
        if self.get_conf('high_dpi_custom_scale_factor', section='main'):
            scale_factor = float(self.get_conf('high_dpi_custom_scale_factors', section='main'))
        else:
            scale_factor = 1
        pm = QPixmap(image_path)
        width = pm.width()
        height = pm.height()
        aspect_ratio = width / height
        width = int(width * 0.8)
        height = int(width / aspect_ratio)
        image = QImage(int(width * scale_factor), int(height * scale_factor), QImage.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        renderer = QSvgRenderer(image_path)
        renderer.render(painter)
        painter.end()
        if scale_factor > 1.0:
            image.setDevicePixelRatio(scale_factor)
        final_pm = QPixmap.fromImage(image)
        final_pm = final_pm.copy(0, 0, int(width * scale_factor), int(height * scale_factor))
        return final_pm

    def focusInEvent(self, event):
        if False:
            return 10
        self._apply_stylesheet(True)
        super().focusOutEvent(event)

    def focusOutEvent(self, event):
        if False:
            print('Hello World!')
        self._apply_stylesheet(False)
        super().focusOutEvent(event)

    def _apply_stylesheet(self, focus):
        if False:
            return 10
        if focus:
            border_color = QStylePalette.COLOR_ACCENT_3
        else:
            border_color = QStylePalette.COLOR_BACKGROUND_4
        qss = qstylizer.style.StyleSheet()
        qss.QFrame.setValues(border=f'1px solid {border_color}', margin='0px', padding='0px', borderRadius=f'{QStylePalette.SIZE_BORDER_RADIUS}')
        self.setStyleSheet(qss.toString())

class HoverRowsTableView(QTableView):
    """
    QTableView subclass that can highlight an entire row when hovered.

    Notes
    -----
    * Classes that inherit from this one need to connect a slot to
      sig_hover_index_changed that handles how the row is painted.
    """
    sig_hover_index_changed = Signal(object)
    '\n    This is emitted when the index that is currently hovered has changed.\n\n    Parameters\n    ----------\n    index: object\n        QModelIndex that has changed on hover.\n    '

    def __init__(self, parent):
        if False:
            return 10
        QTableView.__init__(self, parent)
        self.setMouseTracking(True)
        css = qstylizer.style.StyleSheet()
        css['QTableView::item'].setValues(backgroundColor=f'{QStylePalette.COLOR_BACKGROUND_1}')
        self._stylesheet = css.toString()

    def mouseMoveEvent(self, event):
        if False:
            return 10
        self._inform_hover_index_changed(event)

    def wheelEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super().wheelEvent(event)
        self._inform_hover_index_changed(event)

    def leaveEvent(self, event):
        if False:
            return 10
        super().leaveEvent(event)
        self.setStyleSheet(self._stylesheet)

    def enterEvent(self, event):
        if False:
            print('Hello World!')
        super().enterEvent(event)
        self.setStyleSheet('')

    def _inform_hover_index_changed(self, event):
        if False:
            while True:
                i = 10
        index = self.indexAt(event.pos())
        if index.isValid():
            self.sig_hover_index_changed.emit(index)
            self.viewport().update()

def test_msgcheckbox():
    if False:
        print('Hello World!')
    from spyder.utils.qthelpers import qapplication
    app = qapplication()
    box = MessageCheckBox()
    box.setWindowTitle(_('Spyder updates'))
    box.setText('Testing checkbox')
    box.set_checkbox_text('Check for updates on startup?')
    box.setStandardButtons(QMessageBox.Ok)
    box.setDefaultButton(QMessageBox.Ok)
    box.setIcon(QMessageBox.Information)
    box.exec_()
if __name__ == '__main__':
    test_msgcheckbox()