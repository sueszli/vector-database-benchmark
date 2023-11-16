"""
 @file
 @brief This file contains the project file listview, used by the main window
 @author Noah Figg <eggmunkee@hotmail.com>
 @author Jonathan Thomas <jonathan@openshot.org>

 @section LICENSE

 Copyright (c) 2008-2018 OpenShot Studios, LLC
 (http://www.openshotstudios.com). This file is part of
 OpenShot Video Editor (http://www.openshot.org), an open-source project
 dedicated to delivering high quality video editing and animation solutions
 to the world.

 OpenShot Video Editor is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 OpenShot Video Editor is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with OpenShot Library.  If not, see <http://www.gnu.org/licenses/>.
 """
from PyQt5.QtCore import QSize, Qt, QPoint, QRegExp
from PyQt5.QtGui import QDrag, QCursor
from PyQt5.QtWidgets import QListView, QAbstractItemView, QMenu
from classes import info
from classes.app import get_app
from classes.logger import log
from classes.query import File

class FilesListView(QListView):
    """ A ListView QWidget used on the main window """
    drag_item_size = QSize(48, 48)
    drag_item_center = QPoint(24, 24)

    def contextMenuEvent(self, event):
        if False:
            print('Hello World!')
        event.accept()
        app = get_app()
        app.context_menu_object = 'files'
        index = self.indexAt(event.pos())
        menu = QMenu(self)
        menu.addAction(self.win.actionImportFiles)
        menu.addAction(self.win.actionDetailsView)
        if index.isValid():
            model = self.model()
            id_index = index.sibling(index.row(), 5)
            file_id = model.data(id_index, Qt.DisplayRole)
            menu.addSeparator()
            file = File.get(id=file_id)
            if file and file.data.get('path').endswith('.svg'):
                menu.addAction(self.win.actionEditTitle)
                menu.addAction(self.win.actionDuplicateTitle)
                menu.addSeparator()
            menu.addAction(self.win.actionPreview_File)
            menu.addSeparator()
            menu.addAction(self.win.actionSplitClip)
            menu.addAction(self.win.actionExportClips)
            menu.addSeparator()
            menu.addAction(self.win.actionAdd_to_Timeline)
            menu.addAction(self.win.actionFile_Properties)
            menu.addSeparator()
            menu.addAction(self.win.actionRemove_from_Project)
            menu.addSeparator()
        menu.popup(event.globalPos())

    def dragEnterEvent(self, event):
        if False:
            print('Hello World!')
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        event.accept()
        event.setDropAction(Qt.CopyAction)

    def startDrag(self, supportedActions):
        if False:
            while True:
                i = 10
        ' Override startDrag method to display custom icon '
        selected = self.selectionModel().selectedRows(0)
        current = self.selectionModel().currentIndex()
        if not current.isValid() and selected:
            current = selected[0]
        if not current.isValid():
            log.warning('No draggable items found in model!')
            return False
        icon = current.sibling(current.row(), 0).data(Qt.DecorationRole)
        drag = QDrag(self)
        drag.setMimeData(self.model().mimeData(selected))
        drag.setPixmap(icon.pixmap(self.drag_item_size))
        drag.setHotSpot(self.drag_item_center)
        drag.exec_()

    def dragMoveEvent(self, event):
        if False:
            return 10
        event.accept()

    def dropEvent(self, event):
        if False:
            return 10
        if not event.mimeData().hasUrls():
            event.reject()
            return
        event.accept()
        try:
            get_app().setOverrideCursor(QCursor(Qt.WaitCursor))
            qurl_list = event.mimeData().urls()
            log.info('Processing drop event for {} urls'.format(len(qurl_list)))
            self.files_model.process_urls(qurl_list)
        finally:
            get_app().restoreOverrideCursor()

    def add_file(self, filepath):
        if False:
            while True:
                i = 10
        self.files_model.add_files(filepath)

    def filter_changed(self):
        if False:
            for i in range(10):
                print('nop')
        self.refresh_view()

    def refresh_view(self):
        if False:
            i = 10
            return i + 15
        'Filter files with proxy class'
        model = self.model()
        filter_text = self.win.filesFilter.text()
        model.setFilterRegExp(QRegExp(filter_text.replace(' ', '.*'), Qt.CaseInsensitive))
        col = model.sortColumn()
        model.sort(col)

    def resize_contents(self):
        if False:
            print('Hello World!')
        pass

    def __init__(self, model, *args):
        if False:
            print('Hello World!')
        super().__init__(*args)
        self.win = get_app().window
        self.files_model = model
        self.setModel(self.files_model.proxy_model)
        self.selectionModel().deleteLater()
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionModel(self.files_model.selection_model)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setIconSize(info.LIST_ICON_SIZE)
        self.setGridSize(info.LIST_GRID_SIZE)
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setUniformItemSizes(True)
        self.setStyleSheet('QListView::item { padding-top: 2px; }')
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideRight)
        self.files_model.ModelRefreshed.connect(self.refresh_view)
        app = get_app()
        app.window.filesFilter.textChanged.connect(self.filter_changed)