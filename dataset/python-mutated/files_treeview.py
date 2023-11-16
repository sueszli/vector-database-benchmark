"""
 @file
 @brief This file contains the project file treeview, used by the main window
 @author Noah Figg <eggmunkee@hotmail.com>
 @author Jonathan Thomas <jonathan@openshot.org>
 @author Olivier Girard <eolinwen@gmail.com>

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
import os
from PyQt5.QtCore import QSize, Qt, QPoint
from PyQt5.QtGui import QDrag, QCursor
from PyQt5.QtWidgets import QTreeView, QAbstractItemView, QMenu, QSizePolicy, QHeaderView
from classes import info
from classes.app import get_app
from classes.logger import log
from classes.query import File

class FilesTreeView(QTreeView):
    """ A TreeView QWidget used on the main window """
    drag_item_size = QSize(48, 48)
    drag_item_center = QPoint(24, 24)

    def contextMenuEvent(self, event):
        if False:
            print('Hello World!')
        app = get_app()
        app.context_menu_object = 'files'
        event.accept()
        index = self.indexAt(event.pos())
        menu = QMenu(self)
        menu.addAction(self.win.actionImportFiles)
        menu.addAction(self.win.actionThumbnailView)
        if index.isValid():
            model = index.model()
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
            i = 10
            return i + 15
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()

    def startDrag(self, supportedActions):
        if False:
            return 10
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
            print('Hello World!')
        if not event.mimeData().hasUrls():
            event.ignore()
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
            return 10
        self.files_model.add_files(filepath)

    def filter_changed(self):
        if False:
            return 10
        self.refresh_view()

    def refresh_view(self):
        if False:
            print('Hello World!')
        'Resize and hide certain columns'
        self.hideColumn(3)
        self.hideColumn(4)
        self.hideColumn(5)
        self.resize_contents()

    def resize_contents(self):
        if False:
            for i in range(10):
                print('nop')
        thumbnail_width = 80
        tags_width = 75
        self.header().resizeSection(0, thumbnail_width)
        self.header().resizeSection(2, tags_width)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.header().setSectionResizeMode(2, QHeaderView.Interactive)

    def value_updated(self, item):
        if False:
            while True:
                i = 10
        ' Name or tags updated '
        if self.files_model.ignore_updates:
            return
        _ = get_app()._tr
        file_id = self.files_model.model.item(item.row(), 5).text()
        name = self.files_model.model.item(item.row(), 1).text()
        tags = self.files_model.model.item(item.row(), 2).text()
        f = File.get(id=file_id)
        f.data.update({'name': name or os.path.basename(f.data.get('path'))})
        if 'tags' in f.data or tags:
            f.data.update({'tags': tags})
        f.save()
        self.win.FileUpdated.emit(file_id)

    def __init__(self, model, *args):
        if False:
            i = 10
            return i + 15
        super().__init__(*args)
        self.win = get_app().window
        self.files_model = model
        self.setModel(self.files_model.proxy_model)
        self.selectionModel().deleteLater()
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionModel(self.files_model.selection_model)
        self.setSortingEnabled(True)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setIconSize(info.TREE_ICON_SIZE)
        self.setIndentation(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet('QTreeView::item { padding-top: 2px; }')
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideRight)
        self.files_model.ModelRefreshed.connect(self.refresh_view)