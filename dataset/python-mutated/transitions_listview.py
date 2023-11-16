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
from PyQt5.QtCore import Qt, QSize, QPoint, QRegExp
from PyQt5.QtGui import QDrag
from PyQt5.QtWidgets import QListView, QAbstractItemView, QMenu
from classes import info
from classes.app import get_app
from classes.logger import log

class TransitionsListView(QListView):
    """ A QListView QWidget used on the main window """
    drag_item_size = QSize(48, 48)
    drag_item_center = QPoint(24, 24)

    def contextMenuEvent(self, event):
        if False:
            print('Hello World!')
        event.accept()
        app = get_app()
        app.context_menu_object = 'transitions'
        menu = QMenu(self)
        menu.addAction(self.win.actionDetailsView)
        menu.popup(event.globalPos())

    def startDrag(self, supportedActions):
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

    def filter_changed(self):
        if False:
            print('Hello World!')
        self.refresh_view()

    def refresh_view(self):
        if False:
            while True:
                i = 10
        'Filter transitions with proxy class'
        filter_text = self.win.transitionsFilter.text()
        self.model().setFilterRegExp(QRegExp(filter_text.replace(' ', '.*')))
        self.model().setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.model().sort(Qt.AscendingOrder)

    def __init__(self, model):
        if False:
            while True:
                i = 10
        QListView.__init__(self)
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
        self.setIconSize(info.LIST_ICON_SIZE)
        self.setGridSize(info.LIST_GRID_SIZE)
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setUniformItemSizes(True)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideRight)
        self.setStyleSheet('QListView::item { padding-top: 2px; }')
        app = get_app()
        app.window.transitionsFilter.textChanged.connect(self.filter_changed)
        app.window.refreshTransitionsSignal.connect(self.refresh_view)