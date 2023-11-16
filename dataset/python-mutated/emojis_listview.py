"""
 @file
 @brief This file contains the emojis listview, used by the main window
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
from PyQt5.QtCore import QMimeData, QSize, QPoint, Qt, pyqtSlot, QRegExp
from PyQt5.QtGui import QDrag
from PyQt5.QtWidgets import QListView
import openshot
from classes import info
from classes.query import File
from classes.app import get_app
from classes.logger import log
import json

class EmojisListView(QListView):
    """ A QListView QWidget used on the main window """
    drag_item_size = QSize(48, 48)
    drag_item_center = QPoint(24, 24)

    def dragEnterEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()

    def startDrag(self, event):
        if False:
            print('Hello World!')
        ' Override startDrag method to display custom icon '
        selected = self.selectedIndexes()
        drag = QDrag(self)
        drag.setMimeData(self.model.mimeData(selected))
        icon = self.model.data(selected[0], Qt.DecorationRole)
        drag.setPixmap(icon.pixmap(self.drag_item_size))
        drag.setHotSpot(self.drag_item_center)
        data = json.loads(drag.mimeData().text())
        file = self.add_file(data[0])
        data = QMimeData()
        data.setText(json.dumps([file.id]))
        data.setHtml('clip')
        drag.setMimeData(data)
        drag.exec_()

    def add_file(self, filepath):
        if False:
            return 10
        app = get_app()
        _ = app._tr
        file = File.get(path=filepath)
        if file:
            return file
        clip = openshot.Clip(filepath)
        try:
            reader = clip.Reader()
            file_data = json.loads(reader.Json())
            file_data['media_type'] = 'image'
            file = File()
            file.data = file_data
            file.save()
            return file
        except Exception as ex:
            log.warning('Failed to import file: {}'.format(str(ex)))

    @pyqtSlot(int)
    def group_changed(self, index=-1):
        if False:
            for i in range(10):
                print('nop')
        emoji_group_name = get_app().window.emojiFilterGroup.itemText(index)
        emoji_group_id = get_app().window.emojiFilterGroup.itemData(index)
        self.group_model.setFilterFixedString(emoji_group_id)
        self.group_model.setFilterKeyColumn(2)
        s = get_app().get_settings()
        setting_emoji_group_id = s.get('emoji_group_filter') or 'smileys-emotion'
        if setting_emoji_group_id != emoji_group_id:
            s.set('emoji_group_filter', emoji_group_id)
        self.refresh_view()

    @pyqtSlot(str)
    def filter_changed(self, filter_text=None):
        if False:
            for i in range(10):
                print('nop')
        'Filter emoji with proxy class'
        self.model.setFilterRegExp(QRegExp(filter_text, Qt.CaseInsensitive))
        self.model.setFilterKeyColumn(0)
        self.refresh_view()

    def refresh_view(self):
        if False:
            while True:
                i = 10
        self.model.sort(0)

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        QListView.__init__(self)
        app = get_app()
        _ = app._tr
        self.win = app.window
        self.emojis_model = model
        self.group_model = self.emojis_model.group_model
        self.model = self.emojis_model.proxy_model
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setModel(self.model)
        self.setIconSize(info.EMOJI_ICON_SIZE)
        self.setGridSize(info.EMOJI_GRID_SIZE)
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setUniformItemSizes(True)
        self.setWordWrap(False)
        self.setStyleSheet('QListView::item { padding-top: 2px; }')
        self.refresh_view()
        s = get_app().get_settings()
        default_group_id = s.get('emoji_group_filter') or 'smileys-emotion'
        self.win.emojisFilter.textChanged.connect(self.filter_changed)
        self.win.emojiFilterGroup.clear()
        self.win.emojiFilterGroup.addItem(_('Show All'), '')
        dropdown_index = 0
        for (index, emoji_group_tuple) in enumerate(sorted(self.emojis_model.emoji_groups)):
            (emoji_group_name, emoji_group_id) = emoji_group_tuple
            self.win.emojiFilterGroup.addItem(emoji_group_name, emoji_group_id)
            if emoji_group_id == default_group_id:
                dropdown_index = index + 1
        self.win.emojiFilterGroup.currentIndexChanged.connect(self.group_changed)
        self.win.emojiFilterGroup.setCurrentIndex(dropdown_index)