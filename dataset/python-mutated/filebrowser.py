import os
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QStandardPaths
from picard import log
from picard.config import BoolOption, TextOption, get_config
from picard.const.sys import IS_MACOS
from picard.formats import supported_formats
from picard.util import find_existing_path

def _macos_find_root_volume():
    if False:
        for i in range(10):
            print('nop')
    try:
        for entry in os.scandir('/Volumes/'):
            if entry.is_symlink() and os.path.realpath(entry.path) == '/':
                return entry.path
    except OSError:
        log.warning('Could not detect macOS boot volume', exc_info=True)
    return None

def _macos_extend_root_volume_path(path):
    if False:
        while True:
            i = 10
    if not path.startswith('/Volumes/'):
        root_volume = _macos_find_root_volume()
        if root_volume:
            if path.startswith('/'):
                path = path[1:]
            path = os.path.join(root_volume, path)
    return path
_default_current_browser_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation)
if IS_MACOS:
    _default_current_browser_path = _macos_extend_root_volume_path(_default_current_browser_path)

class FileBrowser(QtWidgets.QTreeView):
    options = [TextOption('persist', 'current_browser_path', _default_current_browser_path), BoolOption('persist', 'show_hidden_files', False)]

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.load_selected_files_action = QtGui.QAction(_('&Load selected files'), self)
        self.load_selected_files_action.triggered.connect(self.load_selected_files)
        self.addAction(self.load_selected_files_action)
        self.move_files_here_action = QtGui.QAction(_('&Move tagged files here'), self)
        self.move_files_here_action.triggered.connect(self.move_files_here)
        self.addAction(self.move_files_here_action)
        self.toggle_hidden_action = QtGui.QAction(_('Show &hidden files'), self)
        self.toggle_hidden_action.setCheckable(True)
        config = get_config()
        self.toggle_hidden_action.setChecked(config.persist['show_hidden_files'])
        self.toggle_hidden_action.toggled.connect(self.show_hidden)
        self.addAction(self.toggle_hidden_action)
        self.set_as_starting_directory_action = QtGui.QAction(_('&Set as starting directory'), self)
        self.set_as_starting_directory_action.triggered.connect(self.set_as_starting_directory)
        self.addAction(self.set_as_starting_directory_action)
        self.doubleClicked.connect(self.load_file_for_item)
        self.focused = False

    def showEvent(self, event):
        if False:
            print('Hello World!')
        if not self.model():
            self._set_model()

    def contextMenuEvent(self, event):
        if False:
            i = 10
            return i + 15
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.load_selected_files_action)
        menu.addSeparator()
        menu.addAction(self.move_files_here_action)
        menu.addAction(self.toggle_hidden_action)
        menu.addAction(self.set_as_starting_directory_action)
        menu.exec(event.globalPos())
        event.accept()

    def _set_model(self):
        if False:
            for i in range(10):
                print('nop')
        model = QtGui.QFileSystemModel()
        self.setModel(model)
        model.layoutChanged.connect(self._layout_changed)
        model.setRootPath('')
        self._set_model_filter()
        filters = []
        for (exts, name) in supported_formats():
            filters.extend(('*' + e for e in exts))
        model.setNameFilters(filters)
        model.setNameFilterDisables(False)
        model.sort(0, QtCore.Qt.SortOrder.AscendingOrder)
        if IS_MACOS:
            self.setRootIndex(model.index('/Volumes'))
        header = self.header()
        header.hideSection(1)
        header.hideSection(2)
        header.hideSection(3)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        header.setVisible(False)

    def _set_model_filter(self):
        if False:
            i = 10
            return i + 15
        config = get_config()
        model_filter = QtCore.QDir.Filter.AllDirs | QtCore.QDir.Filter.Files | QtCore.QDir.Filter.Drives | QtCore.QDir.Filter.NoDotAndDotDot
        if config.persist['show_hidden_files']:
            model_filter |= QtCore.QDir.Filter.Hidden
        self.model().setFilter(model_filter)

    def _layout_changed(self):
        if False:
            print('Hello World!')

        def scroll():
            if False:
                return 10
            if not self.focused:
                self._restore_state()
            self.scrollTo(self.currentIndex())
        QtCore.QTimer.singleShot(0, scroll)

    def scrollTo(self, index, scrolltype=QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible):
        if False:
            return 10
        config = get_config()
        if index and config.setting['filebrowser_horizontal_autoscroll']:
            level = -1
            parent = index.parent()
            root = self.rootIndex()
            while parent.isValid() and parent != root:
                parent = parent.parent()
                level += 1
            pos_x = max(self.indentation() * level, 0)
        else:
            pos_x = self.horizontalScrollBar().value()
        super().scrollTo(index, scrolltype)
        self.horizontalScrollBar().setValue(pos_x)

    def mousePressEvent(self, event):
        if False:
            return 10
        super().mousePressEvent(event)
        index = self.indexAt(event.pos())
        if index.isValid():
            self.selectionModel().setCurrentIndex(index, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)

    def focusInEvent(self, event):
        if False:
            i = 10
            return i + 15
        self.focused = True
        super().focusInEvent(event)

    def show_hidden(self, state):
        if False:
            return 10
        config = get_config()
        config.persist['show_hidden_files'] = state
        self._set_model_filter()

    def save_state(self):
        if False:
            i = 10
            return i + 15
        indexes = self.selectedIndexes()
        if indexes:
            path = self.model().filePath(indexes[0])
            config = get_config()
            config.persist['current_browser_path'] = os.path.normpath(path)

    def restore_state(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _restore_state(self):
        if False:
            print('Hello World!')
        config = get_config()
        if config.setting['starting_directory']:
            path = config.setting['starting_directory_path']
            scrolltype = QtWidgets.QAbstractItemView.ScrollHint.PositionAtTop
        else:
            path = config.persist['current_browser_path']
            scrolltype = QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter
        if path:
            index = self.model().index(find_existing_path(path))
            self.setCurrentIndex(index)
            self.expand(index)
            self.scrollTo(index, scrolltype)

    def _get_destination_from_path(self, path):
        if False:
            return 10
        destination = os.path.normpath(path)
        if not os.path.isdir(destination):
            destination = os.path.dirname(destination)
        return destination

    def load_file_for_item(self, index):
        if False:
            return 10
        model = self.model()
        if not model.isDir(index):
            QtCore.QObject.tagger.add_paths([model.filePath(index)])

    def load_selected_files(self):
        if False:
            return 10
        indexes = self.selectedIndexes()
        if not indexes:
            return
        paths = set((self.model().filePath(index) for index in indexes))
        QtCore.QObject.tagger.add_paths(paths)

    def move_files_here(self):
        if False:
            for i in range(10):
                print('nop')
        indexes = self.selectedIndexes()
        if not indexes:
            return
        config = get_config()
        path = self.model().filePath(indexes[0])
        config.setting['move_files_to'] = self._get_destination_from_path(path)

    def set_as_starting_directory(self):
        if False:
            return 10
        indexes = self.selectedIndexes()
        if indexes:
            config = get_config()
            path = self.model().filePath(indexes[0])
            config.setting['starting_directory_path'] = self._get_destination_from_path(path)