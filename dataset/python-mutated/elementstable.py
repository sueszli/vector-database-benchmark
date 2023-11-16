"""
Table widget to display a set of elements with title, description, icon and an
associated widget.
"""
import sys
from typing import List, Optional, TypedDict
import qstylizer.style
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QAbstractItemView, QCheckBox, QHBoxLayout, QWidget
from superqt.utils import qdebounced
from spyder.api.config.fonts import SpyderFontsMixin, SpyderFontType
from spyder.utils.icon_manager import ima
from spyder.utils.palette import QStylePalette
from spyder.utils.stylesheet import AppStyle
from spyder.widgets.helperwidgets import HoverRowsTableView, HTMLDelegate

class Element(TypedDict):
    """Spec for elements that can be displayed in ElementsTable."""
    title: str
    'Element title'
    description: str
    'Element description'
    additional_info: Optional[str]
    '\n    Additional info that needs to be displayed in a separate column (optional)\n    '
    icon: Optional[QIcon]
    'Element icon (optional)'
    widget: Optional[QWidget]
    '\n    Element widget, e.g. a checkbox or radio button associated to the element\n    (optional)\n    '

class ElementsModel(QAbstractTableModel, SpyderFontsMixin):

    def __init__(self, parent: QWidget, elements: List[Element], with_icons: bool, with_addtional_info: bool, with_widgets: bool):
        if False:
            for i in range(10):
                print('nop')
        QAbstractTableModel.__init__(self)
        self.elements = elements
        self.with_icons = with_icons
        self.n_columns = 1
        self.columns = {'title': 0}
        if with_addtional_info:
            self.n_columns += 1
            self.columns['additional_info'] = 1
        if with_widgets:
            self.n_columns += 1
            if self.n_columns == 3:
                self.columns['widgets'] = 2
            else:
                self.columns['widgets'] = 1
        text_color = QStylePalette.COLOR_TEXT_1
        title_font_size = self.get_font(SpyderFontType.Interface, font_size_delta=1).pointSize()
        self.title_style = f'color:{text_color}; font-size:{title_font_size}pt'
        self.additional_info_style = f'color:{QStylePalette.COLOR_TEXT_4}'
        self.description_style = f'color:{text_color}'

    def data(self, index, role=Qt.DisplayRole):
        if False:
            while True:
                i = 10
        element = self.elements[index.row()]
        if role == Qt.DisplayRole:
            if index.column() == self.columns['title']:
                return self.get_title_repr(element)
            elif index.column() == self.columns.get('additional_info'):
                return self.get_info_repr(element)
            else:
                return None
        elif role == Qt.DecorationRole and self.with_icons:
            if index.column() == self.columns['title']:
                return element['icon']
            else:
                return None
        return None

    def rowCount(self, index=QModelIndex()):
        if False:
            return 10
        return len(self.elements)

    def columnCount(self, index=QModelIndex()):
        if False:
            return 10
        return self.n_columns

    def get_title_repr(self, element: Element) -> str:
        if False:
            i = 10
            return i + 15
        return f'''<table cellspacing="0" cellpadding="3"><tr><td><span style="{self.title_style}">{element['title']}</span></td></tr><tr><td><span style="{self.description_style}">{element['description']}</span></td></tr></table>'''

    def get_info_repr(self, element: Element) -> str:
        if False:
            i = 10
            return i + 15
        if element.get('additional_info'):
            additional_info = f" {element['additional_info']}"
        else:
            return ''
        return f'<span style="{self.additional_info_style}">{additional_info}</span>'

class ElementsTable(HoverRowsTableView):

    def __init__(self, parent: Optional[QWidget], elements: List[Element]):
        if False:
            for i in range(10):
                print('nop')
        HoverRowsTableView.__init__(self, parent)
        self.elements = elements
        self._with_icons = self._with_feature('icon')
        self._with_addtional_info = self._with_feature('additional_info')
        self._with_widgets = self._with_feature('widget')
        self._current_row = -1
        self._current_row_widget = None
        self._is_shown = False
        self.sig_hover_index_changed.connect(self._on_hover_index_changed)
        self.model = ElementsModel(self, self.elements, self._with_icons, self._with_addtional_info, self._with_widgets)
        self.setModel(self.model)
        title_delegate = HTMLDelegate(self, margin=9, wrap_text=True)
        self.setItemDelegateForColumn(self.model.columns['title'], title_delegate)
        self.sig_hover_index_changed.connect(title_delegate.on_hover_index_changed)
        self._info_column_width = 0
        if self._with_addtional_info:
            info_delegate = HTMLDelegate(self, margin=10, align_vcenter=True)
            self.setItemDelegateForColumn(self.model.columns['additional_info'], info_delegate)
            self.sig_hover_index_changed.connect(info_delegate.on_hover_index_changed)
            self.resizeColumnsToContents()
            self._info_column_width = self.horizontalHeader().sectionSize(self.model.columns['additional_info'])
        self._widgets_column_width = 0
        if self._with_widgets:
            widgets_delegate = HTMLDelegate(self, margin=0)
            self.setItemDelegateForColumn(self.model.columns['widgets'], widgets_delegate)
            self.sig_hover_index_changed.connect(widgets_delegate.on_hover_index_changed)
            self.resizeColumnsToContents()
            self._widgets_column_width = self.horizontalHeader().sectionSize(self.model.columns['widgets']) + 15
            for i in range(len(self.elements)):
                layout = QHBoxLayout()
                layout.addWidget(self.elements[i]['widget'])
                layout.setAlignment(Qt.AlignHCenter)
                container_widget = QWidget(self)
                container_widget.setLayout(layout)
                self.elements[i]['row_widget'] = container_widget
                self.setIndexWidget(self.model.index(i, self.model.columns['widgets']), container_widget)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        if self._with_icons:
            self.setIconSize(QSize(32, 32))
        self.setShowGrid(False)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self._set_stylesheet()

    def _on_hover_index_changed(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Actions to take when the index that is hovered has changed.'
        row = index.row()
        if row != self._current_row:
            self._current_row = row
            if self._with_widgets:
                if self._current_row_widget is not None:
                    self._current_row_widget.setStyleSheet('')
                new_row_widget = self.elements[row]['row_widget']
                new_row_widget.setStyleSheet(f'background-color: {QStylePalette.COLOR_BACKGROUND_3}')
                self._current_row_widget = new_row_widget

    def _set_stylesheet(self, leave=False):
        if False:
            i = 10
            return i + 15
        'Set stylesheet when entering or leaving the widget.'
        css = qstylizer.style.StyleSheet()
        bgcolor = QStylePalette.COLOR_BACKGROUND_1 if leave else 'transparent'
        css['QTableView::item'].setValues(borderBottom=f'1px solid {QStylePalette.COLOR_BACKGROUND_4}', paddingLeft='5px', backgroundColor=bgcolor)
        self.setStyleSheet(css.toString())

    def _set_layout(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set rows and columns layout.\n\n        This is necessary to make the table look good at different sizes.\n        '
        extra_width = 0
        if sys.platform == 'darwin':
            if self.verticalScrollBar().isVisible():
                extra_width = AppStyle.MacScrollBarWidth + (15 if self._with_widgets else 5)
            else:
                extra_width = 10 if self._with_widgets else 5
        if self._info_column_width > 0 or self._widgets_column_width > 0:
            title_column_width = self.horizontalHeader().size().width() - (self._info_column_width + self._widgets_column_width + extra_width)
            self.horizontalHeader().resizeSection(self.model.columns['title'], title_column_width)
        self.resizeRowsToContents()
    _set_layout_debounced = qdebounced(_set_layout, timeout=40)
    "\n    Debounced version of _set_layout.\n\n    Notes\n    -----\n    * We need a different version of _set_layout so that we can use the regular\n      one in showEvent. That way users won't experience a visual glitch when\n      the widget is rendered for the first time.\n    * We use this version in resizeEvent, where that is not a problem.\n    "

    def _with_feature(self, feature_name: str) -> bool:
        if False:
            return 10
        "Check if it's necessary to build the table with `feature_name`."
        return len([e for e in self.elements if e.get(feature_name)]) > 0

    def showEvent(self, event):
        if False:
            while True:
                i = 10
        if not self._is_shown:
            self._set_layout()
            self._is_shown = True
        super().showEvent(event)

    def leaveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super().leaveEvent(event)
        if self._current_row_widget is not None:
            self._current_row_widget.setStyleSheet('')
        self._set_stylesheet(leave=True)

    def enterEvent(self, event):
        if False:
            i = 10
            return i + 15
        super().enterEvent(event)
        if self._current_row_widget is not None:
            self._current_row_widget.setStyleSheet(f'background-color: {QStylePalette.COLOR_BACKGROUND_3}')
        self._set_stylesheet()

    def resizeEvent(self, event):
        if False:
            return 10
        self._set_layout_debounced()
        super().resizeEvent(event)

def test_elements_table():
    if False:
        for i in range(10):
            print('nop')
    from spyder.utils.qthelpers import qapplication
    app = qapplication()
    elements_with_title = [{'title': 'IPython console', 'description': 'Execute code'}, {'title': 'Help', 'description': 'Look for help'}]
    table = ElementsTable(None, elements_with_title)
    table.show()
    elements_with_icons = [{'title': 'IPython console', 'description': 'Execute code', 'icon': ima.icon('ipython_console')}, {'title': 'Help', 'description': 'Look for help', 'icon': ima.icon('help')}]
    table_with_icons = ElementsTable(None, elements_with_icons)
    table_with_icons.show()
    elements_with_widgets = [{'title': 'IPython console', 'description': 'Execute code', 'icon': ima.icon('ipython_console'), 'widget': QCheckBox()}, {'title': 'Help', 'description': 'Look for help', 'icon': ima.icon('help'), 'widget': QCheckBox()}]
    table_with_widgets = ElementsTable(None, elements_with_widgets)
    table_with_widgets.show()
    elements_with_info = [{'title': 'IPython console', 'description': 'Execute code', 'icon': ima.icon('ipython_console'), 'widget': QCheckBox(), 'additional_info': 'Core plugin'}, {'title': 'Help', 'description': 'Look for help', 'icon': ima.icon('help'), 'widget': QCheckBox()}]
    table_with_widgets_and_icons = ElementsTable(None, elements_with_info)
    table_with_widgets_and_icons.show()
    app.exec_()
if __name__ == '__main__':
    test_elements_table()