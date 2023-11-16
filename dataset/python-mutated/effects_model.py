"""
 @file
 @brief This file contains the effects model, used by the main window
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
import os
from PyQt5.QtCore import QObject, QMimeData, Qt, QSize, pyqtSignal, QSortFilterProxyModel, QPersistentModelIndex, QItemSelectionModel
from PyQt5.QtGui import QIcon, QPixmap, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QMessageBox
import openshot
from classes import info
from classes.logger import log
from classes.app import get_app
import json

class EffectsProxyModel(QSortFilterProxyModel):

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent=parent)

    def filterAcceptsRow(self, sourceRow, sourceParent):
        if False:
            i = 10
            return i + 15
        'Filter for common transitions and text filter'
        if not get_app().window.actionEffectsShowAll.isChecked():
            effect_name = self.sourceModel().data(self.sourceModel().index(sourceRow, 1, sourceParent))
            effect_desc = self.sourceModel().data(self.sourceModel().index(sourceRow, 2, sourceParent))
            effect_type = self.sourceModel().data(self.sourceModel().index(sourceRow, 3, sourceParent))
            if get_app().window.actionEffectsShowVideo.isChecked():
                return effect_type == 'Video' and self.filterRegExp().indexIn(effect_name) >= 0 and (self.filterRegExp().indexIn(effect_desc) >= 0)
            else:
                return effect_type == 'Audio' and self.filterRegExp().indexIn(effect_name) >= 0 and (self.filterRegExp().indexIn(effect_desc) >= 0)
        return super(EffectsProxyModel, self).filterAcceptsRow(sourceRow, sourceParent)

    def mimeData(self, indexes):
        if False:
            while True:
                i = 10
        data = QMimeData()
        items = [i.sibling(i.row(), 4).data() for i in indexes]
        data.setText(json.dumps(items))
        data.setHtml('effect')
        return data

class EffectsModel(QObject):
    ModelRefreshed = pyqtSignal()

    def update_model(self, clear=True):
        if False:
            for i in range(10):
                print('nop')
        log.info('updating effects model.')
        app = get_app()
        win = app.window
        _ = app._tr
        if clear:
            self.model_names = {}
            self.model.clear()
        self.model.setHorizontalHeaderLabels([_('Thumb'), _('Name'), _('Description')])
        effects_dir = os.path.join(info.PATH, 'effects')
        icons_dir = os.path.join(effects_dir, 'icons')
        raw_effects_list = json.loads(openshot.EffectInfo.Json())
        for effect_info in raw_effects_list:
            effect_name = effect_info['class_name']
            title = effect_info['name']
            description = effect_info['description']
            icon_name = '%s.png' % effect_name.lower().replace(' ', '')
            icon_path = os.path.join(icons_dir, icon_name)
            category = None
            if effect_info['has_video'] and effect_info['has_audio']:
                category = 'Audio & Video'
            elif not effect_info['has_video'] and effect_info['has_audio']:
                category = 'Audio'
            elif effect_info['has_video'] and (not effect_info['has_audio']):
                category = 'Video'
            if win.effectsFilter.text() != '' and win.effectsFilter.text().lower() not in self.app._tr(title).lower() and (win.effectsFilter.text().lower() not in self.app._tr(description).lower()):
                continue
            thumb_path = os.path.join(info.IMAGES_PATH, 'cache', icon_name)
            if not os.path.exists(thumb_path):
                thumb_path = os.path.join(info.CACHE_PATH, icon_name)
            if not os.path.exists(thumb_path):
                try:
                    log.info('Generating thumbnail for %s (%s)' % (thumb_path, icon_path))
                    clip = openshot.Clip(icon_path)
                    reader = clip.Reader()
                    reader.Open()
                    reader.GetFrame(0).Thumbnail(thumb_path, 98, 64, os.path.join(info.IMAGES_PATH, 'mask.png'), '', '#000', True, 'png', 85)
                    reader.Close()
                except Exception:
                    log.info('Invalid effect image file: %s' % icon_path)
                    msg = QMessageBox()
                    msg.setText(_('{} is not a valid image file.'.format(icon_path)))
                    msg.exec_()
                    continue
            row = []
            col = QStandardItem()
            icon = QIcon()
            icon.addFile(thumb_path)
            col.setIcon(icon)
            col.setText(self.app._tr(title))
            col.setToolTip(self.app._tr(title))
            col.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            row.append(col)
            col = QStandardItem('Name')
            col.setData(self.app._tr(title), Qt.DisplayRole)
            col.setText(self.app._tr(title))
            col.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            row.append(col)
            col = QStandardItem('Description')
            col.setData(self.app._tr(description), Qt.DisplayRole)
            col.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            row.append(col)
            col = QStandardItem('Category')
            col.setData(category, Qt.DisplayRole)
            col.setText(category)
            col.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            row.append(col)
            col = QStandardItem('Effect')
            col.setData(effect_name, Qt.DisplayRole)
            col.setText(effect_name)
            col.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            row.append(col)
            if effect_name not in self.model_names:
                self.model.appendRow(row)
                self.model_names[effect_name] = QPersistentModelIndex(row[1].index())
        self.ModelRefreshed.emit()

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        super().__init__(*args)
        self.app = get_app()
        self.model = QStandardItemModel()
        self.model.setColumnCount(5)
        self.model_names = {}
        self.proxy_model = EffectsProxyModel()
        self.proxy_model.setDynamicSortFilter(False)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy_model.setSortCaseSensitivity(Qt.CaseSensitive)
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setSortLocaleAware(True)
        self.selection_model = QItemSelectionModel(self.proxy_model)
        if info.MODEL_TEST:
            try:
                from PyQt5.QtTest import QAbstractItemModelTester
                self.model_tests = []
                for m in [self.proxy_model, self.model]:
                    self.model_tests.append(QAbstractItemModelTester(m, QAbstractItemModelTester.FailureReportingMode.Warning))
                log.info('Enabled {} model tests for effects data'.format(len(self.model_tests)))
            except ImportError:
                pass