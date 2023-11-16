"""Results browser."""
import os.path as osp
from qtpy.QtCore import QPoint, QSize, Qt, Signal, Slot
from qtpy.QtGui import QAbstractTextDocumentLayout, QColor, QFontMetrics, QTextDocument
from qtpy.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QTreeWidgetItem
from spyder.api.config.fonts import SpyderFontsMixin, SpyderFontType
from spyder.api.translations import _
from spyder.plugins.findinfiles.widgets.search_thread import ELLIPSIS, MAX_RESULT_LENGTH
from spyder.utils import icon_manager as ima
from spyder.utils.palette import QStylePalette
from spyder.widgets.onecolumntree import OneColumnTree
ON = 'on'
OFF = 'off'

class LineMatchItem(QTreeWidgetItem):

    def __init__(self, parent, lineno, colno, match, font, text_color):
        if False:
            print('Hello World!')
        self.lineno = lineno
        self.colno = colno
        self.match = match['formatted_text']
        self.plain_match = match['text']
        self.text_color = text_color
        self.font = font
        super().__init__(parent, [self.__repr__()], QTreeWidgetItem.Type)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        match = str(self.match).rstrip()
        _str = f"""<!-- LineMatchItem --><p style="color:'{self.text_color}';">&nbsp;&nbsp;<b>{self.lineno}</b> ({self.colno}): <span style='font-family:{self.font.family()};font-size:{self.font.pointSize()}pt;'>{match}</span></p>"""
        return _str

    def __unicode__(self):
        if False:
            return 10
        return self.__repr__()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.__repr__()

    def __lt__(self, x):
        if False:
            return 10
        return self.lineno < x.lineno

    def __ge__(self, x):
        if False:
            return 10
        return self.lineno >= x.lineno

class FileMatchItem(QTreeWidgetItem):

    def __init__(self, parent, path, filename, sorting, text_color):
        if False:
            return 10
        self.sorting = sorting
        self.filename = osp.basename(filename)
        dirname = osp.dirname(filename)
        try:
            rel_dirname = dirname.split(path)[1]
            if rel_dirname.startswith(osp.sep):
                rel_dirname = rel_dirname[1:]
        except IndexError:
            rel_dirname = dirname
        self.rel_dirname = rel_dirname
        title = f'<!-- FileMatchItem --><b style="color:{text_color}">{osp.basename(filename)}</b>&nbsp;&nbsp;&nbsp;<span style="color:{text_color}"><em>{self.rel_dirname}</em></span>'
        super().__init__(parent, [title], QTreeWidgetItem.Type)
        self.setIcon(0, ima.get_icon_by_extension_or_type(filename, 1.0))
        self.setToolTip(0, filename)

    def __lt__(self, x):
        if False:
            return 10
        if self.sorting['status'] == ON:
            return self.filename < x.filename
        else:
            return False

    def __ge__(self, x):
        if False:
            i = 10
            return i + 15
        if self.sorting['status'] == ON:
            return self.filename >= x.filename
        else:
            return False

class ItemDelegate(QStyledItemDelegate):

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._margin = None
        self._background_color = QColor(QStylePalette.COLOR_BACKGROUND_3)
        self.width = 0

    def paint(self, painter, option, index):
        if False:
            i = 10
            return i + 15
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        style = QApplication.style() if options.widget is None else options.widget.style()
        if options.state & QStyle.State_MouseOver:
            painter.fillRect(option.rect, self._background_color)
        doc = QTextDocument()
        text = options.text
        doc.setHtml(text)
        doc.setDocumentMargin(0)
        options.text = ''
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)
        ctx = QAbstractTextDocumentLayout.PaintContext()
        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options, None)
        painter.save()
        painter.translate(textRect.topLeft() + QPoint(0, 4))
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        if False:
            for i in range(10):
                print('nop')
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        doc = QTextDocument()
        doc.setHtml(options.text)
        doc.setTextWidth(options.rect.width())
        size = QSize(self.width, int(doc.size().height()))
        return size

class ResultsBrowser(OneColumnTree, SpyderFontsMixin):
    sig_edit_goto_requested = Signal(str, int, str, int, int)
    sig_max_results_reached = Signal()

    def __init__(self, parent, text_color, max_results=1000):
        if False:
            return 10
        super().__init__(parent)
        self.search_text = None
        self.results = None
        self.max_results = max_results
        self.total_matches = None
        self.error_flag = None
        self.completed = None
        self.sorting = {}
        self.font = self.get_font(SpyderFontType.MonospaceInterface)
        self.data = None
        self.files = None
        self.root_items = None
        self.text_color = text_color
        self.path = None
        self.longest_file_item = ''
        self.longest_line_item = ''
        self.set_title('')
        self.set_sorting(OFF)
        self.setSortingEnabled(False)
        self.setItemDelegate(ItemDelegate(self))
        self.setUniformRowHeights(True)
        self.sortByColumn(0, Qt.AscendingOrder)
        self.common_actions = self.common_actions[:2]
        self.header().sectionClicked.connect(self.sort_section)

    def activated(self, item):
        if False:
            print('Hello World!')
        'Double-click event.'
        itemdata = self.data.get(id(self.currentItem()))
        if itemdata is not None:
            (filename, lineno, colno, colend) = itemdata
            self.sig_edit_goto_requested.emit(filename, lineno, self.search_text, colno, colend - colno)

    def set_sorting(self, flag):
        if False:
            i = 10
            return i + 15
        'Enable result sorting after search is complete.'
        self.sorting['status'] = flag
        self.header().setSectionsClickable(flag == ON)

    @Slot(int)
    def sort_section(self, idx):
        if False:
            return 10
        self.setSortingEnabled(True)

    def clicked(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Click event.'
        if isinstance(item, FileMatchItem):
            if item.isExpanded():
                self.collapseItem(item)
            else:
                self.expandItem(item)
        else:
            self.activated(item)

    def clear_title(self, search_text):
        if False:
            i = 10
            return i + 15
        self.font = self.get_font(SpyderFontType.MonospaceInterface)
        self.clear()
        self.setSortingEnabled(False)
        self.num_files = 0
        self.data = {}
        self.files = {}
        self.set_sorting(OFF)
        self.search_text = search_text
        title = "'%s' - " % search_text
        text = _('String not found')
        self.set_title(title + text)

    @Slot(object)
    def append_file_result(self, filename):
        if False:
            while True:
                i = 10
        'Real-time update of file items.'
        if len(self.data) < self.max_results:
            item = FileMatchItem(self, self.path, filename, self.sorting, self.text_color)
            self.files[filename] = item
            item.setExpanded(True)
            self.num_files += 1
            item_text = osp.join(item.rel_dirname, item.filename)
            if len(item_text) > len(self.longest_file_item):
                self.longest_file_item = item_text

    @Slot(object, object)
    def append_result(self, items, title):
        if False:
            return 10
        'Real-time update of line items.'
        if len(self.data) >= self.max_results:
            self.set_title(_('Maximum number of results reached! Try narrowing the search.'))
            self.sig_max_results_reached.emit()
            return
        available = self.max_results - len(self.data)
        if available < len(items):
            items = items[:available]
        self.setUpdatesEnabled(False)
        self.set_title(title)
        for item in items:
            (filename, lineno, colno, line, match_end) = item
            file_item = self.files.get(filename, None)
            if file_item:
                item = LineMatchItem(file_item, lineno, colno, line, self.font, self.text_color)
                self.data[id(item)] = (filename, lineno, colno, match_end)
                if len(item.plain_match) > len(self.longest_line_item):
                    self.longest_line_item = item.plain_match
        self.setUpdatesEnabled(True)

    def set_max_results(self, value):
        if False:
            while True:
                i = 10
        'Set maximum amount of results to add.'
        self.max_results = value

    def set_path(self, path):
        if False:
            print('Hello World!')
        'Set path where the search is performed.'
        self.path = path

    def set_width(self):
        if False:
            while True:
                i = 10
        'Set widget width according to its longest item.'
        file_item_size = self.fontMetrics().size(Qt.TextSingleLine, self.longest_file_item)
        file_item_width = file_item_size.width()
        metrics = QFontMetrics(self.font)
        line_item_chars = len(self.longest_line_item)
        if line_item_chars >= MAX_RESULT_LENGTH:
            line_item_chars = MAX_RESULT_LENGTH + len(ELLIPSIS) + 1
        line_item_width = line_item_chars * metrics.width('W')
        if file_item_width > line_item_width:
            width = file_item_width
        else:
            width = line_item_width
        self.itemDelegate().width = width + 10