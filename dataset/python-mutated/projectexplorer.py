"""Project Explorer"""
import os
import os.path as osp
import shutil
from qtpy.QtCore import QSortFilterProxyModel, Qt, Signal, Slot
from qtpy.QtWidgets import QAbstractItemView, QHeaderView, QMessageBox
from spyder.api.translations import _
from spyder.py3compat import to_text_string
from spyder.utils import misc
from spyder.plugins.explorer.widgets.explorer import DirView

class ProxyModel(QSortFilterProxyModel):
    """Proxy model to filter tree view."""
    PATHS_TO_HIDE = ['.spyproject', '__pycache__', '.ipynb_checkpoints', '.git', '.hg', '.svn', '.pytest_cache', '.DS_Store', 'Thumbs.db', '.directory']
    PATHS_TO_SHOW = ['.github']

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        'Initialize the proxy model.'
        super(ProxyModel, self).__init__(parent)
        self.root_path = None
        self.path_list = []
        self.setDynamicSortFilter(True)

    def setup_filter(self, root_path, path_list):
        if False:
            print('Hello World!')
        '\n        Setup proxy model filter parameters.\n\n        Parameters\n        ----------\n        root_path: str\n            Root path of the proxy model.\n        path_list: list\n            List with all the paths.\n        '
        self.root_path = osp.normpath(str(root_path))
        self.path_list = [osp.normpath(str(p)) for p in path_list]
        self.invalidateFilter()

    def sort(self, column, order=Qt.AscendingOrder):
        if False:
            print('Hello World!')
        'Reimplement Qt method.'
        self.sourceModel().sort(column, order)

    def filterAcceptsRow(self, row, parent_index):
        if False:
            print('Hello World!')
        'Reimplement Qt method.'
        if self.root_path is None:
            return True
        index = self.sourceModel().index(row, 0, parent_index)
        path = osp.normcase(osp.normpath(str(self.sourceModel().filePath(index))))
        if osp.normcase(self.root_path).startswith(path):
            return True
        else:
            for p in [osp.normcase(p) for p in self.path_list]:
                if path == p or path.startswith(p + os.sep):
                    if not any([path.endswith(os.sep + d) for d in self.PATHS_TO_SHOW]):
                        if any([path.endswith(os.sep + d) for d in self.PATHS_TO_HIDE]):
                            return False
                        else:
                            return True
                    else:
                        return True
            else:
                return False

    def data(self, index, role):
        if False:
            return 10
        'Show tooltip with full path only for the root directory.'
        if role == Qt.ToolTipRole:
            root_dir = self.path_list[0].split(osp.sep)[-1]
            if index.data() == root_dir:
                return osp.join(self.root_path, root_dir)
        return QSortFilterProxyModel.data(self, index, role)

    def type(self, index):
        if False:
            while True:
                i = 10
        '\n        Returns the type of file for the given index.\n\n        Parameters\n        ----------\n        index: int\n            Given index to search its type.\n        '
        return self.sourceModel().type(self.mapToSource(index))

class FilteredDirView(DirView):
    """Filtered file/directory tree view."""

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        'Initialize the filtered dir view.'
        super().__init__(parent)
        self.proxymodel = None
        self.setup_proxy_model()
        self.root_path = None

    def setup_proxy_model(self):
        if False:
            i = 10
            return i + 15
        'Setup proxy model.'
        self.proxymodel = ProxyModel(self)
        self.proxymodel.setSourceModel(self.fsmodel)

    def install_model(self):
        if False:
            while True:
                i = 10
        'Install proxy model.'
        if self.root_path is not None:
            self.setModel(self.proxymodel)

    def set_root_path(self, root_path):
        if False:
            i = 10
            return i + 15
        '\n        Set root path.\n\n        Parameters\n        ----------\n        root_path: str\n            New path directory.\n        '
        self.root_path = root_path
        self.install_model()
        index = self.fsmodel.setRootPath(root_path)
        self.proxymodel.setup_filter(self.root_path, [])
        self.setRootIndex(self.proxymodel.mapFromSource(index))

    def get_index(self, filename):
        if False:
            while True:
                i = 10
        '\n        Return index associated with filename.\n\n        Parameters\n        ----------\n        filename: str\n            String with the filename.\n        '
        index = self.fsmodel.index(filename)
        if index.isValid() and index.model() is self.fsmodel:
            return self.proxymodel.mapFromSource(index)

    def set_folder_names(self, folder_names):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set folder names\n\n        Parameters\n        ----------\n        folder_names: list\n            List with the folder names.\n        '
        assert self.root_path is not None
        path_list = [osp.join(self.root_path, dirname) for dirname in folder_names]
        self.proxymodel.setup_filter(self.root_path, path_list)

    def get_filename(self, index):
        if False:
            i = 10
            return i + 15
        '\n        Return filename from index\n\n        Parameters\n        ----------\n        index: int\n            Index of the list of filenames\n        '
        if index:
            path = self.fsmodel.filePath(self.proxymodel.mapToSource(index))
            return osp.normpath(str(path))

    def setup_project_view(self):
        if False:
            i = 10
            return i + 15
        'Setup view for projects.'
        for i in [1, 2, 3]:
            self.hideColumn(i)
        self.setHeaderHidden(True)

    def directory_clicked(self, dirname, index):
        if False:
            while True:
                i = 10
        if index and index.isValid():
            if self.get_conf('single_click_to_open'):
                state = not self.isExpanded(index)
            else:
                state = self.isExpanded(index)
            self.setExpanded(index, state)

class ProjectExplorerTreeWidget(FilteredDirView):
    """Explorer tree widget"""
    sig_delete_project = Signal()

    def __init__(self, parent, show_hscrollbar=True):
        if False:
            i = 10
            return i + 15
        FilteredDirView.__init__(self, parent)
        self.last_folder = None
        self.setSelectionMode(FilteredDirView.ExtendedSelection)
        self.show_hscrollbar = show_hscrollbar
        self.setDragEnabled(True)
        self.setDragDropMode(FilteredDirView.DragDrop)

    @Slot(bool)
    def toggle_hscrollbar(self, checked):
        if False:
            for i in range(10):
                print('nop')
        'Toggle horizontal scrollbar'
        self.set_conf('show_hscrollbar', checked)
        self.show_hscrollbar = checked
        self.header().setStretchLastSection(not checked)
        self.header().setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)

    def dragMoveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Reimplement Qt method'
        index = self.indexAt(event.pos())
        if index:
            dst = self.get_filename(index)
            if osp.isdir(dst):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        if False:
            while True:
                i = 10
        'Reimplement Qt method'
        event.ignore()
        action = event.dropAction()
        if action not in (Qt.MoveAction, Qt.CopyAction):
            return
        dst = self.get_filename(self.indexAt(event.pos()))
        (yes_to_all, no_to_all) = (None, None)
        src_list = [to_text_string(url.toString()) for url in event.mimeData().urls()]
        if len(src_list) > 1:
            buttons = QMessageBox.Yes | QMessageBox.YesToAll | QMessageBox.No | QMessageBox.NoToAll | QMessageBox.Cancel
        else:
            buttons = QMessageBox.Yes | QMessageBox.No
        for src in src_list:
            if src == dst:
                continue
            dst_fname = osp.join(dst, osp.basename(src))
            if osp.exists(dst_fname):
                if yes_to_all is not None or no_to_all is not None:
                    if no_to_all:
                        continue
                elif osp.isfile(dst_fname):
                    answer = QMessageBox.warning(self, _('Project explorer'), _('File <b>%s</b> already exists.<br>Do you want to overwrite it?') % dst_fname, buttons)
                    if answer == QMessageBox.No:
                        continue
                    elif answer == QMessageBox.Cancel:
                        break
                    elif answer == QMessageBox.YesToAll:
                        yes_to_all = True
                    elif answer == QMessageBox.NoToAll:
                        no_to_all = True
                        continue
                else:
                    QMessageBox.critical(self, _('Project explorer'), _('Folder <b>%s</b> already exists.') % dst_fname, QMessageBox.Ok)
                    event.setDropAction(Qt.CopyAction)
                    return
            try:
                if action == Qt.CopyAction:
                    if osp.isfile(src):
                        shutil.copy(src, dst)
                    else:
                        shutil.copytree(src, dst)
                else:
                    if osp.isfile(src):
                        misc.move_file(src, dst)
                    else:
                        shutil.move(src, dst)
                    self.parent_widget.removed.emit(src)
            except EnvironmentError as error:
                if action == Qt.CopyAction:
                    action_str = _('copy')
                else:
                    action_str = _('move')
                QMessageBox.critical(self, _('Project Explorer'), _('<b>Unable to %s <i>%s</i></b><br><br>Error message:<br>%s') % (action_str, src, str(error)))

    @Slot()
    def delete(self, fnames=None):
        if False:
            while True:
                i = 10
        'Delete files'
        if fnames is None:
            fnames = self.get_selected_filenames()
        multiple = len(fnames) > 1
        yes_to_all = None
        for fname in fnames:
            if fname == self.proxymodel.path_list[0]:
                self.sig_delete_project.emit()
            else:
                yes_to_all = self.delete_file(fname, multiple, yes_to_all)
                if yes_to_all is not None and (not yes_to_all):
                    break