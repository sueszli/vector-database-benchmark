"""
 @file
 @brief This file contains the transitions file treeview, used by the main window
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
from PyQt5.QtCore import Qt, QSize, QPoint
from PyQt5.QtGui import QDrag
from PyQt5.QtWidgets import QTreeView, QAbstractItemView, QMenu, QSizePolicy
from classes import info
from classes.app import get_app
from classes.logger import log

class TransitionsTreeView(QTreeView):
    """ A TreeView QWidget used on the main window """
    drag_item_size = QSize(48, 48)
    drag_item_center = QPoint(24, 24)

    def contextMenuEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        app = get_app()
        app.context_menu_object = 'transitions'
        menu = QMenu(self)
        menu.addAction(self.win.actionThumbnailView)
        menu.popup(event.globalPos())

    def startDrag(self, event):
        if False:
            for i in range(10):
                print('nop')
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

    def refresh_columns(self):
        if False:
            while True:
                i = 10
        'Hide certain columns'
        if type(self) == TransitionsTreeView:
            self.hideColumn(2)
            self.hideColumn(3)
            self.setColumnWidth(0, 80)
        self.sortByColumn(1, Qt.AscendingOrder)

    def __init__(self, model):
        if False:
            print('Hello World!')
        QTreeView.__init__(self)
        self.win = get_app().window
        self.transition_model = model
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setModel(self.transition_model.proxy_model)
        self.selectionModel().deleteLater()
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionModel(self.transition_model.selection_model)
        self.setSortingEnabled(True)
        self.setIconSize(info.TREE_ICON_SIZE)
        self.setIndentation(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setWordWrap(True)
        self.setStyleSheet('QTreeView::item { padding-top: 2px; }')
        self.transition_model.ModelRefreshed.connect(self.refresh_columns)
        self.refresh_columns()