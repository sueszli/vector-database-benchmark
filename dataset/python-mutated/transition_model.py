"""
 @file
 @brief This file contains the transitions model, used by the main window
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
from PyQt5.QtCore import QObject, QMimeData, Qt, pyqtSignal, QSortFilterProxyModel, QPersistentModelIndex, QItemSelectionModel
from PyQt5.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QMessageBox
import openshot
from classes import info
from classes.logger import log
from classes.app import get_app
import json

class TransitionFilterProxyModel(QSortFilterProxyModel):
    """Proxy class used for sorting and filtering model data"""

    def filterAcceptsRow(self, sourceRow, sourceParent):
        if False:
            while True:
                i = 10
        'Filter for common transitions and text filter'
        if get_app().window.actionTransitionsShowCommon.isChecked():
            index = self.sourceModel().index(sourceRow, 2, sourceParent)
            group_name = self.sourceModel().data(index)
            index = self.sourceModel().index(sourceRow, 0, sourceParent)
            trans_name = self.sourceModel().data(index)
            return group_name == 'common' and self.filterRegExp().indexIn(trans_name) >= 0
        return super(TransitionFilterProxyModel, self).filterAcceptsRow(sourceRow, sourceParent)

    def lessThan(self, left, right):
        if False:
            return 10
        'Sort with both group name and transition name'
        leftData = left.data(self.sortRole())
        rightData = right.data(self.sortRole())
        leftGroup = left.sibling(left.row(), 2).data()
        rightGroup = right.sibling(right.row(), 2).data()
        return leftGroup <= rightGroup and leftData < rightData

    def mimeData(self, indexes):
        if False:
            for i in range(10):
                print('nop')
        data = QMimeData()
        items = [i.sibling(i.row(), 3).data() for i in indexes]
        data.setText(json.dumps(items))
        data.setHtml('transition')
        return data

class TransitionsModel(QObject):
    ModelRefreshed = pyqtSignal()

    def update_model(self, clear=True):
        if False:
            return 10
        log.info('updating transitions model.')
        app = get_app()
        _ = app._tr
        if clear:
            self.model_paths = {}
            self.model.clear()
        self.model.setHorizontalHeaderLabels([_('Thumb'), _('Name')])
        transitions_dir = os.path.join(info.PATH, 'transitions')
        common_dir = os.path.join(transitions_dir, 'common')
        extra_dir = os.path.join(transitions_dir, 'extra')
        transition_groups = [{'type': 'common', 'dir': common_dir, 'files': os.listdir(common_dir)}, {'type': 'extra', 'dir': extra_dir, 'files': os.listdir(extra_dir)}]
        if os.path.exists(info.TRANSITIONS_PATH) and os.listdir(info.TRANSITIONS_PATH):
            transition_groups.append({'type': 'user', 'dir': info.TRANSITIONS_PATH, 'files': os.listdir(info.TRANSITIONS_PATH)})
        for group in transition_groups:
            type = group['type']
            dir = group['dir']
            files = group['files']
            for filename in sorted(files):
                path = os.path.join(dir, filename)
                fileBaseName = os.path.splitext(filename)[0]
                if filename[0] == '.' or 'thumbs.db' in filename.lower():
                    continue
                suffix_number = None
                name_parts = fileBaseName.split('_')
                if name_parts[-1].isdigit():
                    suffix_number = name_parts[-1]
                trans_name = fileBaseName.replace('_', ' ').capitalize()
                if suffix_number:
                    trans_name = trans_name.replace(suffix_number, '%s')
                    trans_name = self.app._tr(trans_name) % suffix_number
                else:
                    trans_name = self.app._tr(trans_name)
                thumb_path = os.path.join(info.IMAGES_PATH, 'cache', '{}.png'.format(fileBaseName))
                if not os.path.exists(thumb_path):
                    thumb_path = os.path.join(info.CACHE_PATH, '{}.png'.format(fileBaseName))
                if not os.path.exists(thumb_path):
                    try:
                        clip = openshot.Clip(path)
                        reader = clip.Reader()
                        reader.Open()
                        reader.GetFrame(0).Thumbnail(thumb_path, 98, 64, os.path.join(info.IMAGES_PATH, 'mask.png'), '', '#000', True, 'png', 85)
                        reader.Close()
                        clip.Close()
                    except Exception:
                        log.debug('Invalid transition image file %s', filename, exc_info=1)
                        msg = QMessageBox()
                        msg.setText(_('{} is not a valid transition file.'.format(filename)))
                        msg.exec_()
                        continue
                row = []
                icon = QIcon()
                icon.addFile(thumb_path)
                col = QStandardItem()
                col.setIcon(icon)
                col.setText(trans_name)
                col.setToolTip(trans_name)
                col.setData(type)
                col.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
                row.append(col)
                col = QStandardItem('Name')
                col.setData(trans_name, Qt.DisplayRole)
                col.setText(trans_name)
                col.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
                row.append(col)
                col = QStandardItem('Type')
                col.setData(type, Qt.DisplayRole)
                col.setText(type)
                col.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
                row.append(col)
                col = QStandardItem('Path')
                col.setData(path, Qt.DisplayRole)
                col.setText(path)
                col.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
                row.append(col)
                if path not in self.model_paths:
                    self.model.appendRow(row)
                    self.model_paths[path] = QPersistentModelIndex(row[3].index())
        self.ModelRefreshed.emit()

    def __init__(self, *args):
        if False:
            print('Hello World!')
        super().__init__(*args)
        self.app = get_app()
        self.model = QStandardItemModel()
        self.model.setColumnCount(4)
        self.model_paths = {}
        self.proxy_model = TransitionFilterProxyModel()
        self.proxy_model.setDynamicSortFilter(True)
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
                log.info('Enabled {} model tests for transition data'.format(len(self.model_tests)))
            except ImportError:
                pass