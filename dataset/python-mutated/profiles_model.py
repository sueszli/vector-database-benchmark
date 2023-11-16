"""
 @file
 @brief This file contains the profiles model, used by the Profile dialog
 @author Jonathan Thomas <jonathan@openshot.org>

 @section LICENSE

 Copyright (c) 2008-2023 OpenShot Studios, LLC
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
from PyQt5.QtCore import Qt, QSortFilterProxyModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from classes.logger import log
from classes.app import get_app

class ProfilesProxyModel(QSortFilterProxyModel):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)

    def filterAcceptsRow(self, sourceRow, sourceParent):
        if False:
            i = 10
            return i + 15
        'Filter for common transitions and text filter'
        profile_key = self.sourceModel().data(self.sourceModel().index(sourceRow, 0, sourceParent))
        profile_desc = self.sourceModel().data(self.sourceModel().index(sourceRow, 1, sourceParent))
        profile_dar = self.sourceModel().data(self.sourceModel().index(sourceRow, 5, sourceParent))
        return self.filterRegExp().indexIn(profile_key.lower()) >= 0 or self.filterRegExp().indexIn(profile_desc.lower()) >= 0 or self.filterRegExp().indexIn(profile_dar) >= 0

class ProfilesStandardItemModel(QStandardItemModel):

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        QStandardItemModel.__init__(self)

class ProfilesModel:

    def update_model(self, filter=None, clear=True):
        if False:
            i = 10
            return i + 15
        log.debug('updating profiles model.')
        app = get_app()
        _ = app._tr
        if clear:
            log.debug('cleared profiles model')
            self.model.clear()
        self.model.setHorizontalHeaderLabels([_('Key'), _('Description'), _('Width'), _('Height'), _('FPS'), _('DAR'), _('SAR')])
        for profile in self.profiles_list:
            if filter and (not (filter.lower() in profile.info.description.lower() or filter.lower() in f'{profile.info.width}x{profile.info.height}' or filter.lower() in f'{profile.info.display_ratio.num}:{profile.info.display_ratio.den}')):
                continue
            row = []
            flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
            item = QStandardItem(f'{profile.Key()}')
            item.setData(profile, Qt.UserRole)
            row.append(item)
            item = QStandardItem(f'{profile.info.description}')
            item.setFlags(flags)
            row.append(item)
            item = QStandardItem(f'{profile.info.width}')
            item.setFlags(flags)
            row.append(item)
            item = QStandardItem(f'{profile.info.height}')
            item.setFlags(flags)
            row.append(item)
            fps_string = f'{profile.info.fps.num / profile.info.fps.den:.2f}'
            if profile.info.fps.den == 1:
                fps_string = f'{int(profile.info.fps.num / profile.info.fps.den)}'
            item = QStandardItem(fps_string)
            item.setFlags(flags)
            row.append(item)
            item = QStandardItem(f'{profile.info.display_ratio.num}:{profile.info.display_ratio.den}')
            item.setFlags(flags)
            row.append(item)
            item = QStandardItem(f'{profile.info.pixel_ratio.num}:{profile.info.pixel_ratio.den}')
            item.setFlags(flags)
            row.append(item)
            self.model.appendRow(row)

    def __init__(self, profiles, *args):
        if False:
            for i in range(10):
                print('nop')
        _ = get_app()._tr
        self.app = get_app()
        self.model = ProfilesStandardItemModel()
        self.model.setColumnCount(6)
        self.profiles_list = profiles
        self.proxy_model = ProfilesProxyModel()
        self.proxy_model.setDynamicSortFilter(False)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy_model.setSortCaseSensitivity(Qt.CaseSensitive)
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setSortLocaleAware(True)