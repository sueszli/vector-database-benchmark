"""The ListView to display downloads in."""
import functools
from typing import Callable, MutableSequence, Tuple, Union
from qutebrowser.qt.core import pyqtSlot, QSize, Qt
from qutebrowser.qt.widgets import QListView, QSizePolicy, QMenu, QStyleFactory
from qutebrowser.browser import downloads
from qutebrowser.config import stylesheet
from qutebrowser.utils import qtutils, utils
_ActionListType = MutableSequence[Union[Tuple[None, None], Tuple[str, Callable[[], None]]]]

class DownloadView(QListView):
    """QListView which shows currently running downloads as a bar.

    Attributes:
        _menu: The QMenu which is currently displayed.
    """
    STYLESHEET = '\n        QListView {\n            background-color: {{ conf.colors.downloads.bar.bg }};\n            font: {{ conf.fonts.downloads }};\n            border: 0;\n        }\n\n        QListView::item {\n            padding-right: 2px;\n        }\n    '

    def __init__(self, model, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        if not utils.is_mac:
            self.setStyle(QStyleFactory.create('Fusion'))
        stylesheet.set_register(self)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setSpacing(1)
        self._menu = None
        model.rowsInserted.connect(self._update_geometry)
        model.rowsRemoved.connect(self._update_geometry)
        model.dataChanged.connect(self._update_geometry)
        self.setModel(model)
        self.setWrapping(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.clicked.connect(self.on_clicked)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        model = qtutils.add_optional(self.model())
        count: Union[int, str]
        if model is None:
            count = 'None'
        else:
            count = model.rowCount()
        return utils.get_repr(self, count=count)

    def _model(self) -> downloads.DownloadModel:
        if False:
            return 10
        'Get the current download model.\n\n        Ensures the model is not None.\n        '
        model = self.model()
        assert isinstance(model, downloads.DownloadModel), model
        return model

    @pyqtSlot()
    def _update_geometry(self):
        if False:
            return 10
        'Wrapper to call updateGeometry.\n\n        For some reason, this is needed so that PyQt disconnects the signals and handles\n        arguments correctly. Probably a WORKAROUND for an unknown PyQt bug.\n        '
        self.updateGeometry()

    @pyqtSlot(bool)
    def on_fullscreen_requested(self, on):
        if False:
            while True:
                i = 10
        'Hide/show the downloadview when entering/leaving fullscreen.'
        if on:
            self.hide()
        else:
            self.show()

    @pyqtSlot('QModelIndex')
    def on_clicked(self, index):
        if False:
            print('Hello World!')
        'Handle clicking of an item.\n\n        Args:\n            index: The QModelIndex of the clicked item.\n        '
        if not index.isValid():
            return
        item = self._model().data(index, downloads.ModelRole.item)
        if item.done and item.successful:
            item.open_file()
            item.remove()

    def _get_menu_actions(self, item: downloads.AbstractDownloadItem) -> _ActionListType:
        if False:
            for i in range(10):
                print('nop')
        'Get the available context menu actions for a given DownloadItem.\n\n        Args:\n            item: The DownloadItem to get the actions for, or None.\n        '
        model = self._model()
        actions: _ActionListType = []
        if item is None:
            pass
        elif item.done:
            if item.successful:
                actions.append(('Open', item.open_file))
                actions.append(('Open directory', functools.partial(item.open_file, open_dir=True, cmdline=None)))
            else:
                actions.append(('Retry', item.try_retry))
            actions.append(('Remove', item.remove))
        else:
            actions.append(('Cancel', item.cancel))
        if item is not None:
            actions.append(('Copy URL', functools.partial(utils.set_clipboard, item.url().toDisplayString())))
        if model.can_clear():
            actions.append((None, None))
            actions.append(('Remove all finished', model.download_clear))
        return actions

    @pyqtSlot('QPoint')
    def show_context_menu(self, point):
        if False:
            return 10
        'Show the context menu.'
        index = self.indexAt(point)
        if index.isValid():
            item = self._model().data(index, downloads.ModelRole.item)
        else:
            item = None
        self._menu = QMenu(self)
        actions = self._get_menu_actions(item)
        for (name, handler) in actions:
            if name is None and handler is None:
                self._menu.addSeparator()
            else:
                assert name is not None
                assert handler is not None
                action = self._menu.addAction(name)
                assert action is not None
                action.triggered.connect(handler)
        if actions:
            viewport = self.viewport()
            assert viewport is not None
            self._menu.popup(viewport.mapToGlobal(point))

    def minimumSizeHint(self):
        if False:
            i = 10
            return i + 15
        'Override minimumSizeHint so the size is correct in a layout.'
        return self.sizeHint()

    def sizeHint(self):
        if False:
            while True:
                i = 10
        'Return sizeHint based on the view contents.'
        idx = self._model().last_index()
        bottom = self.visualRect(idx).bottom()
        if bottom != -1:
            margins = self.contentsMargins()
            height = bottom + margins.top() + margins.bottom() + 2 * self.spacing()
            size = QSize(0, height)
        else:
            size = QSize(0, 0)
        qtutils.ensure_valid(size)
        return size