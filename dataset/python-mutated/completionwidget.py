"""Completion view for statusbar command section.

Defines a CompletionView which uses CompletionFiterModel and CompletionModel
subclasses to provide completions.
"""
from typing import TYPE_CHECKING, Optional
from qutebrowser.qt.widgets import QTreeView, QSizePolicy, QStyleFactory, QWidget
from qutebrowser.qt.core import pyqtSlot, pyqtSignal, Qt, QItemSelectionModel, QSize
from qutebrowser.config import config, stylesheet
from qutebrowser.completion import completiondelegate
from qutebrowser.completion.models import completionmodel
from qutebrowser.utils import utils, usertypes, debug, log, qtutils
from qutebrowser.api import cmdutils
if TYPE_CHECKING:
    from qutebrowser.mainwindow.statusbar import command

class CompletionView(QTreeView):
    """The view showing available completions.

    Based on QTreeView but heavily customized so root elements show as category
    headers, and children show as flat list.

    Attributes:
        pattern: Current filter pattern, used for highlighting.
        _win_id: The ID of the window this CompletionView is associated with.
        _height: The height to use for the CompletionView.
        _height_perc: Either None or a percentage if height should be relative.
        _delegate: The item delegate used.
        _column_widths: A list of column widths, in percent.
        _active: Whether a selection is active.
        _cmd: The statusbar Command object.

    Signals:
        update_geometry: Emitted when the completion should be resized.
        selection_changed: Emitted when the completion item selection changes.
    """
    STYLESHEET = '\n        QTreeView {\n            font: {{ conf.fonts.completion.entry }};\n            background-color: {{ conf.colors.completion.even.bg }};\n            alternate-background-color: {{ conf.colors.completion.odd.bg }};\n            outline: 0;\n            border: 0px;\n        }\n\n        QTreeView::item:disabled {\n            background-color: {{ conf.colors.completion.category.bg }};\n            border-top: 1px solid\n                {{ conf.colors.completion.category.border.top }};\n            border-bottom: 1px solid\n                {{ conf.colors.completion.category.border.bottom }};\n        }\n\n        QTreeView::item:selected, QTreeView::item:selected:hover {\n            border-top: 1px solid\n                {{ conf.colors.completion.item.selected.border.top }};\n            border-bottom: 1px solid\n                {{ conf.colors.completion.item.selected.border.bottom }};\n            background-color: {{ conf.colors.completion.item.selected.bg }};\n        }\n\n        QTreeView:item::hover {\n            border: 0px;\n        }\n\n        QTreeView QScrollBar {\n            width: {{ conf.completion.scrollbar.width }}px;\n            background: {{ conf.colors.completion.scrollbar.bg }};\n        }\n\n        QTreeView QScrollBar::handle {\n            background: {{ conf.colors.completion.scrollbar.fg }};\n            border: {{ conf.completion.scrollbar.padding }}px solid\n                    {{ conf.colors.completion.scrollbar.bg }};\n            min-height: 10px;\n        }\n\n        QTreeView QScrollBar::sub-line, QScrollBar::add-line {\n            border: none;\n            background: none;\n        }\n    '
    update_geometry = pyqtSignal()
    selection_changed = pyqtSignal(str)

    def __init__(self, *, cmd: 'command.Command', win_id: int, parent: QWidget=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.pattern: Optional[str] = None
        self._win_id = win_id
        self._cmd = cmd
        self._active = False
        config.instance.changed.connect(self._on_config_changed)
        self._delegate = completiondelegate.CompletionItemDelegate(self)
        self.setItemDelegate(self._delegate)
        self.setStyle(QStyleFactory.create('Fusion'))
        stylesheet.set_register(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setHeaderHidden(True)
        self.setAlternatingRowColors(True)
        self.setIndentation(0)
        self.setItemsExpandable(False)
        self.setExpandsOnDoubleClick(False)
        self.setAnimated(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setUniformRowHeights(True)
        self.hide()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return utils.get_repr(self)

    def _model(self) -> completionmodel.CompletionModel:
        if False:
            return 10
        'Get the current completion model.\n\n        Ensures the model is not None.\n        '
        model = self.model()
        assert isinstance(model, completionmodel.CompletionModel), model
        return model

    def _selection_model(self) -> QItemSelectionModel:
        if False:
            return 10
        'Get the current selection model.\n\n        Ensures the model is not None.\n        '
        model = self.selectionModel()
        assert model is not None
        return model

    @pyqtSlot(str)
    def _on_config_changed(self, option):
        if False:
            return 10
        if option in ['completion.height', 'completion.shrink']:
            self.update_geometry.emit()

    def _resize_columns(self):
        if False:
            for i in range(10):
                print('nop')
        'Resize the completion columns based on column_widths.'
        if self.model() is None:
            return
        width = self.size().width()
        column_widths = self._model().column_widths
        pixel_widths = [width * perc // 100 for perc in column_widths]
        bar = self.verticalScrollBar()
        assert bar is not None
        delta = bar.sizeHint().width()
        for (i, width) in reversed(list(enumerate(pixel_widths))):
            if width > delta:
                pixel_widths[i] -= delta
                break
        for (i, w) in enumerate(pixel_widths):
            assert w >= 0, (i, w)
            self.setColumnWidth(i, w)

    def _next_idx(self, upwards):
        if False:
            while True:
                i = 10
        'Get the previous/next QModelIndex displayed in the view.\n\n        Used by tab_handler.\n\n        Args:\n            upwards: Get previous item, not next.\n\n        Return:\n            A QModelIndex.\n        '
        model = self._model()
        idx = self._selection_model().currentIndex()
        if not idx.isValid():
            if upwards:
                return model.last_item()
            else:
                return model.first_item()
        while True:
            idx = self.indexAbove(idx) if upwards else self.indexBelow(idx)
            if not idx.isValid() and upwards:
                return model.last_item()
            elif not idx.isValid() and (not upwards):
                idx = model.first_item()
                self.scrollTo(idx.parent())
                return idx
            elif idx.parent().isValid():
                return idx
        raise utils.Unreachable

    def _next_page(self, upwards):
        if False:
            i = 10
            return i + 15
        'Return the index a page away from the selected index.\n\n        Args:\n            upwards: Get previous item, not next.\n\n        Return:\n            A QModelIndex.\n        '
        old_idx = self._selection_model().currentIndex()
        idx = old_idx
        model = self._model()
        if not idx.isValid():
            return model.last_item() if upwards else model.first_item()
        rect = self.visualRect(idx)
        qtutils.ensure_valid(rect)
        page_length = self.height() // rect.height()
        offset = -(page_length - 1) if upwards else page_length - 1
        idx = model.sibling(old_idx.row() + offset, old_idx.column(), old_idx)
        while idx.isValid() and (not idx.parent().isValid()):
            idx = self.indexAbove(idx) if upwards else self.indexBelow(idx)
        if idx.isValid():
            return idx
        border_item = model.first_item() if upwards else model.last_item()
        if old_idx == border_item:
            return self._next_idx(upwards)
        if upwards:
            self.scrollTo(border_item.parent())
        return border_item

    def _next_category_idx(self, upwards):
        if False:
            print('Hello World!')
        'Get the index of the previous/next category.\n\n        Args:\n            upwards: Get previous item, not next.\n\n        Return:\n            A QModelIndex.\n        '
        idx = self._selection_model().currentIndex()
        model = self._model()
        if not idx.isValid():
            return self._next_idx(upwards).sibling(0, 0)
        idx = idx.parent()
        direction = -1 if upwards else 1
        while True:
            idx = idx.sibling(idx.row() + direction, 0)
            if idx.isValid():
                child = model.index(0, 0, idx)
                if child.isValid():
                    self.scrollTo(idx)
                    return child
            elif upwards:
                return model.last_item().sibling(0, 0)
            else:
                idx = model.first_item()
                self.scrollTo(idx.parent())
                return idx
        raise utils.Unreachable

    @cmdutils.register(instance='completion', modes=[usertypes.KeyMode.command], scope='window')
    @cmdutils.argument('which', choices=['next', 'prev', 'next-category', 'prev-category', 'next-page', 'prev-page'])
    @cmdutils.argument('history', flag='H')
    def completion_item_focus(self, which, history=False):
        if False:
            while True:
                i = 10
        "Shift the focus of the completion menu to another item.\n\n        Args:\n            which: 'next', 'prev',\n                   'next-category', 'prev-category',\n                   'next-page', or 'prev-page'.\n            history: Navigate through command history if no text was typed.\n        "
        if history:
            if self._cmd.text() == ':' or self._cmd.history.is_browsing() or (not self._active):
                if which == 'next':
                    self._cmd.command_history_next()
                    return
                elif which == 'prev':
                    self._cmd.command_history_prev()
                    return
                else:
                    raise cmdutils.CommandError("Can't combine --history with {}!".format(which))
        if not self._active:
            return
        selmodel = self._selection_model()
        indices = {'next': lambda : self._next_idx(upwards=False), 'prev': lambda : self._next_idx(upwards=True), 'next-category': lambda : self._next_category_idx(upwards=False), 'prev-category': lambda : self._next_category_idx(upwards=True), 'next-page': lambda : self._next_page(upwards=False), 'prev-page': lambda : self._next_page(upwards=True)}
        idx = indices[which]()
        if not idx.isValid():
            return
        selmodel.setCurrentIndex(idx, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
        next_idx = self.indexBelow(idx)
        if not self.visualRect(next_idx).isValid():
            self.expandAll()
        count = self._model().count()
        if count == 0:
            self.hide()
        elif count == 1 and config.val.completion.quick:
            self.hide()
        elif config.val.completion.show == 'auto':
            self.show()

    def set_model(self, model):
        if False:
            print('Hello World!')
        'Switch completion to a new model.\n\n        Called from on_update_completion().\n\n        Args:\n            model: The model to use.\n        '
        old_model = self.model()
        if old_model is not None and model is not old_model:
            old_model.deleteLater()
            self._selection_model().deleteLater()
        self.setModel(model)
        if model is None:
            self._active = False
            self.hide()
            return
        model.setParent(self)
        self._active = True
        self.pattern = None
        self._maybe_show()
        self._resize_columns()
        for i in range(model.rowCount()):
            self.expand(model.index(i, 0))

    def set_pattern(self, pattern: str) -> None:
        if False:
            while True:
                i = 10
        'Set the pattern on the underlying model.'
        if not self.model():
            return
        if self.pattern == pattern:
            log.completion.debug('Ignoring pattern set request as pattern has not changed.')
            return
        self.pattern = pattern
        with debug.log_time(log.completion, 'Set pattern {}'.format(pattern)):
            self._model().set_pattern(pattern)
            self._selection_model().clear()
            self._maybe_update_geometry()
            self._maybe_show()

    def _maybe_show(self):
        if False:
            for i in range(10):
                print('nop')
        if config.val.completion.show == 'always' and self._model().count() > 0:
            self.show()
        else:
            self.hide()

    def _maybe_update_geometry(self):
        if False:
            return 10
        'Emit the update_geometry signal if the config says so.'
        if config.val.completion.shrink:
            self.update_geometry.emit()

    @pyqtSlot()
    def on_clear_completion_selection(self):
        if False:
            return 10
        'Clear the selection model when an item is activated.'
        self.hide()
        selmod = self._selection_model()
        if selmod is not None:
            selmod.clearSelection()
            selmod.clearCurrentIndex()

    def sizeHint(self):
        if False:
            i = 10
            return i + 15
        'Get the completion size according to the config.'
        confheight = str(config.val.completion.height)
        if confheight.endswith('%'):
            perc = int(confheight.rstrip('%'))
            window = self.window()
            assert window is not None
            height = window.height() * perc // 100
        else:
            height = int(confheight)
        if config.val.completion.shrink:
            bar = self.horizontalScrollBar()
            assert bar is not None
            contents_height = self.viewportSizeHint().height() + bar.sizeHint().height()
            if contents_height <= height:
                height = contents_height
        return QSize(-1, height)

    def selectionChanged(self, selected, deselected):
        if False:
            for i in range(10):
                print('nop')
        'Extend selectionChanged to call completers selection_changed.'
        if not self._active:
            return
        super().selectionChanged(selected, deselected)
        indexes = selected.indexes()
        if not indexes:
            return
        data = str(self._model().data(indexes[0]))
        self.selection_changed.emit(data)

    def resizeEvent(self, e):
        if False:
            i = 10
            return i + 15
        'Extend resizeEvent to adjust column size.'
        super().resizeEvent(e)
        self._resize_columns()

    def showEvent(self, e):
        if False:
            print('Hello World!')
        "Adjust the completion size and scroll when it's freshly shown."
        self.update_geometry.emit()
        scrollbar = self.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.minimum())
        super().showEvent(e)

    @cmdutils.register(instance='completion', modes=[usertypes.KeyMode.command], scope='window')
    def completion_item_del(self):
        if False:
            while True:
                i = 10
        'Delete the current completion item.'
        index = self.currentIndex()
        if not index.isValid():
            raise cmdutils.CommandError('No item selected!')
        self._model().delete_cur_item(index)

    @cmdutils.register(instance='completion', modes=[usertypes.KeyMode.command], scope='window')
    def completion_item_yank(self, sel=False):
        if False:
            print('Hello World!')
        'Yank the current completion item into the clipboard.\n\n        Args:\n            sel: Use the primary selection instead of the clipboard.\n        '
        text = self._cmd.selectedText()
        if not text:
            index = self.currentIndex()
            if not index.isValid():
                raise cmdutils.CommandError('No item selected!')
            text = self._model().data(index)
        if not utils.supports_selection():
            sel = False
        utils.set_clipboard(text, selection=sel)