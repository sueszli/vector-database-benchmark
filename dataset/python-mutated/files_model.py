"""
 @file
 @brief This file contains the project file model, used by the project tree
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
import os
import json
import re
import glob
import functools
from PyQt5.QtCore import QMimeData, Qt, pyqtSignal, QEventLoop, QObject, QSortFilterProxyModel, QItemSelectionModel, QPersistentModelIndex
from PyQt5.QtGui import QIcon, QStandardItem, QStandardItemModel
from classes import updates
from classes import info
from classes.image_types import get_media_type
from classes.query import File
from classes.logger import log
from classes.app import get_app
from classes.thumbnail import GetThumbPath
import openshot

class FileFilterProxyModel(QSortFilterProxyModel):
    """Proxy class used for sorting and filtering model data"""

    def filterAcceptsRow(self, sourceRow, sourceParent):
        if False:
            i = 10
            return i + 15
        'Filter for text'
        if get_app().window.actionFilesShowVideo.isChecked() or get_app().window.actionFilesShowAudio.isChecked() or get_app().window.actionFilesShowImage.isChecked() or get_app().window.filesFilter.text():
            index = self.sourceModel().index(sourceRow, 0, sourceParent)
            file_name = self.sourceModel().data(index)
            index = self.sourceModel().index(sourceRow, 3, sourceParent)
            media_type = self.sourceModel().data(index)
            index = self.sourceModel().index(sourceRow, 2, sourceParent)
            tags = self.sourceModel().data(index)
            if any([get_app().window.actionFilesShowVideo.isChecked() and media_type != 'video', get_app().window.actionFilesShowAudio.isChecked() and media_type != 'audio', get_app().window.actionFilesShowImage.isChecked() and media_type != 'image']):
                return False
            return self.filterRegExp().indexIn(file_name) >= 0 or self.filterRegExp().indexIn(tags) >= 0
        return super().filterAcceptsRow(sourceRow, sourceParent)

    def mimeData(self, indexes):
        if False:
            return 10
        data = QMimeData()
        ids = self.parent.selected_file_ids()
        data.setText(json.dumps(ids))
        data.setHtml('clip')
        return data

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'parent' in kwargs:
            self.parent = kwargs['parent']
            kwargs.pop('parent')
        super().__init__(**kwargs)

class FilesModel(QObject, updates.UpdateInterface):
    ModelRefreshed = pyqtSignal()

    def changed(self, action):
        if False:
            while True:
                i = 10
        if len(action.key) >= 1 and action.key[0].lower() == 'files' or action.type == 'load':
            if action.type == 'insert':
                self.update_model(clear=False)
            elif action.type == 'delete' and action.key[0].lower() == 'files':
                self.update_model(clear=False, delete_file_id=action.key[1].get('id', ''))
            elif action.type == 'update' and action.key[0].lower() == 'files':
                self.update_model(clear=False, update_file_id=action.key[1].get('id', ''))
            else:
                self.update_model(clear=True)

    def update_model(self, clear=True, delete_file_id=None, update_file_id=None):
        if False:
            i = 10
            return i + 15
        log.debug('updating files model.')
        app = get_app()
        self.ignore_updates = True
        _ = app._tr
        if delete_file_id in self.model_ids:
            id_index = self.model_ids[delete_file_id]
            if not id_index.isValid() or delete_file_id != id_index.data():
                log.warning("Couldn't remove {} from model!".format(delete_file_id))
                return
            row_num = id_index.row()
            self.model.removeRows(row_num, 1, id_index.parent())
            self.model.submit()
            self.model_ids.pop(delete_file_id)
        if update_file_id in self.model_ids:
            id_index = self.model_ids[update_file_id]
            if not id_index.isValid() or update_file_id != id_index.data():
                log.warning("Couldn't update {} in model!".format(update_file_id))
                return
            f = File.get(id=update_file_id)
            if f:
                row_num = id_index.row()
                if f.data.get('tags') != self.model.item(row_num, 2).text():
                    self.model.item(row_num, 2).setText(f.data.get('tags'))
        if clear:
            self.model_ids = {}
            self.model.clear()
        self.model.setHorizontalHeaderLabels(['', _('Name'), _('Tags')])
        files = File.filter()
        row_added_count = 0
        for file in files:
            id = file.data['id']
            if id in self.model_ids and self.model_ids[id].isValid():
                continue
            (path, filename) = os.path.split(file.data['path'])
            tags = file.data.get('tags', '')
            name = file.data.get('name', filename)
            media_type = file.data.get('media_type')
            if media_type in ['video', 'image']:
                thumbnail_frame = 1
                if 'start' in file.data:
                    fps = file.data['fps']
                    fps_float = float(fps['num']) / float(fps['den'])
                    thumbnail_frame = round(float(file.data['start']) * fps_float) + 1
                thumb_icon = QIcon(GetThumbPath(file.id, thumbnail_frame))
            else:
                thumb_icon = QIcon(os.path.join(info.PATH, 'images', 'AudioThumbnail.svg'))
            row = []
            flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled | Qt.ItemNeverHasChildren
            col = QStandardItem(thumb_icon, name)
            col.setToolTip(filename)
            col.setFlags(flags)
            row.append(col)
            col = QStandardItem(name)
            col.setFlags(flags | Qt.ItemIsEditable)
            row.append(col)
            col = QStandardItem(tags)
            col.setFlags(flags | Qt.ItemIsEditable)
            row.append(col)
            col = QStandardItem(media_type)
            col.setFlags(flags)
            row.append(col)
            col = QStandardItem(path)
            col.setFlags(flags)
            row.append(col)
            col = QStandardItem(id)
            col.setFlags(flags | Qt.ItemIsUserCheckable)
            row.append(col)
            if id not in self.model_ids:
                self.model.appendRow(row)
                self.model_ids[id] = QPersistentModelIndex(row[5].index())
                row_added_count += 1
                if row_added_count % 2 == 0:
                    get_app().processEvents(QEventLoop.ExcludeUserInputEvents)
            get_app().window.resize_contents()
        self.ignore_updates = False
        self.ModelRefreshed.emit()

    def add_files(self, files, image_seq_details=None, quiet=False, prevent_image_seq=False, prevent_recent_folder=False):
        if False:
            for i in range(10):
                print('nop')
        app = get_app()
        settings = app.get_settings()
        _ = app._tr
        if not isinstance(files, (list, tuple)):
            files = [files]
        start_count = len(files)
        for (count, filepath) in enumerate(files):
            (dir_path, filename) = os.path.split(filepath)
            new_file = File.get(path=filepath)
            if new_file:
                del new_file
                continue
            try:
                clip = openshot.Clip(filepath)
                reader = clip.Reader()
                file_data = json.loads(reader.Json())
                file_data['media_type'] = get_media_type(file_data)
                if file_data.get('has_audio') and (not file_data.get('has_video')):
                    project = get_app().project
                    file_data['width'] = project.get('width')
                    file_data['height'] = project.get('height')
                new_file = File()
                new_file.data = file_data
                seq_info = None
                if not prevent_image_seq:
                    seq_info = image_seq_details or self.get_image_sequence_details(filepath)
                if seq_info:
                    new_path = seq_info.get('path')
                    clip = openshot.Clip(new_path)
                    new_file.data = json.loads(clip.Reader().Json())
                    if clip and clip.info.duration > 0.0:
                        new_file.data['media_type'] = 'video'
                        duration = new_file.data['duration']
                        if seq_info and 'fps' in seq_info and ('length_multiplier' in seq_info):
                            fps_num = seq_info.get('fps', {}).get('num', 25)
                            fps_den = seq_info.get('fps', {}).get('den', 1)
                            log.debug('Image Sequence using specified FPS: %s / %s' % (fps_num, fps_den))
                        else:
                            fps_num = get_app().project.get('fps').get('num', 30)
                            fps_den = get_app().project.get('fps').get('den', 1)
                            log.debug('Image Sequence using project FPS: %s / %s' % (fps_num, fps_den))
                        duration *= 25.0 / (float(fps_num) / float(fps_den))
                        new_file.data['duration'] = duration
                        new_file.data['fps'] = {'num': fps_num, 'den': fps_den}
                        new_file.data['video_timebase'] = {'num': fps_den, 'den': fps_num}
                        log.info(f"Imported '{new_path}' as image sequence with '{fps_num}/{fps_den}' FPS and '{duration}' duration")
                        match_glob = '{}{}.{}'.format(seq_info.get('base_name'), '[0-9]*', seq_info.get('extension'))
                        log.debug('Removing files from import list with glob: {}'.format(match_glob))
                        for seq_file in glob.iglob(os.path.join(seq_info.get('folder_path'), match_glob)):
                            if seq_file in files and seq_file != filepath:
                                files.remove(seq_file)
                    else:
                        log.info(f'Failed to parse image sequence pattern {new_path}, ignoring...')
                        continue
                if not seq_info:
                    log.info('Imported media file {}'.format(filepath))
                new_file.save()
                if start_count > 15:
                    message = _('Importing %(count)d / %(total)d') % {'count': count, 'total': len(files) - 1}
                    app.window.statusBar.showMessage(message, 15000)
                get_app().processEvents()
                if not prevent_recent_folder:
                    settings.setDefaultPath(settings.actionType.IMPORT, dir_path)
            except Exception as ex:
                log.warning('Failed to import {}: {}'.format(filepath, ex))
                if not quiet:
                    app.window.invalidImage(filename)
        self.ignore_image_sequence_paths = []
        message = _('Imported %(count)d files') % {'count': len(files) - 1}
        app.window.statusBar.showMessage(message, 3000)

    def get_image_sequence_details(self, file_path):
        if False:
            while True:
                i = 10
        'Inspect a file path and determine if this is an image sequence'
        (dirName, fileName) = os.path.split(file_path)
        if dirName in self.ignore_image_sequence_paths:
            return None
        extensions = ['png', 'jpg', 'jpeg', 'gif', 'tif', 'svg']
        match = re.findall('(.*[^\\d])?(0*)(\\d+)\\.(%s)' % '|'.join(extensions), fileName, re.I)
        if not match:
            return None
        base_name = match[0][0]
        fixlen = match[0][1] > ''
        number = int(match[0][2])
        digits = len(match[0][1] + match[0][2])
        extension = match[0][3]
        full_base_name = os.path.join(dirName, base_name)
        fixlen = fixlen or not (glob.glob('%s%s.%s' % (full_base_name, '[0-9]' * (digits + 1), extension)) or glob.glob('%s%s.%s' % (full_base_name, '[0-9]' * (digits - 1 if digits > 1 else 3), extension)))
        for x in range(max(0, number - 100), min(number + 101, 50000)):
            if x != number and os.path.exists('%s%s.%s' % (full_base_name, str(x).rjust(digits, '0') if fixlen else str(x), extension)):
                break
        else:
            return None
        log.debug('Ignoring path for image sequence imports: {}'.format(dirName))
        self.ignore_image_sequence_paths.append(dirName)
        log.info('Prompt user to import sequence starting from {}'.format(fileName))
        if not get_app().window.promptImageSequence(fileName):
            return None
        if not fixlen:
            zero_pattern = '%d'
        else:
            zero_pattern = '%%0%sd' % digits
        pattern = '%s%s.%s' % (base_name, zero_pattern, extension)
        new_file_path = os.path.join(dirName, pattern)
        parameters = {'folder_path': dirName, 'base_name': base_name, 'fixlen': fixlen, 'digits': digits, 'extension': extension, 'pattern': pattern, 'path': new_file_path}
        return parameters

    def process_urls(self, qurl_list):
        if False:
            while True:
                i = 10
        'Recursively process QUrls from a QDropEvent'
        import_quietly = False
        media_paths = []
        for uri in qurl_list:
            filepath = uri.toLocalFile()
            if not os.path.exists(filepath):
                continue
            if filepath.endswith('.osp') and os.path.isfile(filepath):
                get_app().window.OpenProjectSignal.emit(filepath)
                return True
            if os.path.isdir(filepath):
                import_quietly = True
                log.info('Recursively importing {}'.format(filepath))
                try:
                    for (r, _, f) in os.walk(filepath):
                        media_paths.extend([os.path.join(r, p) for p in f])
                except OSError:
                    log.warning('Directory recursion failed', exc_info=1)
            elif os.path.isfile(filepath):
                media_paths.append(filepath)
        if not media_paths:
            return
        media_paths.sort()
        log.debug('Importing file list: {}'.format(media_paths))
        self.add_files(media_paths, quiet=import_quietly)

    def update_file_thumbnail(self, file_id):
        if False:
            return 10
        'Update/re-generate the thumbnail of a specific file'
        file = File.get(id=file_id)
        (path, filename) = os.path.split(file.data['path'])
        name = file.data.get('name', filename)
        fps = file.data['fps']
        fps_float = float(fps['num']) / float(fps['den'])
        self.ignore_updates = True
        m = self.model
        if file_id in self.model_ids:
            id_index = self.model_ids[file_id]
            if not id_index.isValid():
                return
            if file.data.get('media_type') in ['video', 'image']:
                thumbnail_frame = 1
                if 'start' in file.data:
                    thumbnail_frame = round(float(file.data['start']) * fps_float) + 1
                thumb_icon = QIcon(GetThumbPath(file.id, thumbnail_frame, clear_cache=True))
            else:
                thumb_icon = QIcon(os.path.join(info.PATH, 'images', 'AudioThumbnail.svg'))
            thumb_index = id_index.sibling(id_index.row(), 0)
            item = m.itemFromIndex(thumb_index)
            item.setIcon(thumb_icon)
            item.setText(name)
            self.ModelRefreshed.emit()
        self.ignore_updates = False

    def selected_file_ids(self):
        if False:
            return 10
        ' Get a list of file IDs for all selected files '
        selected = self.selection_model.selectedRows(5)
        return [idx.data() for idx in selected]

    def selected_files(self):
        if False:
            return 10
        ' Get a list of File objects representing the current selection '
        files = []
        for id in self.selected_file_ids():
            files.append(File.get(id=id))
        return files

    def current_file_id(self):
        if False:
            for i in range(10):
                print('nop')
        ' Get the file ID of the current files-view item, or the first selection '
        cur = self.selection_model.currentIndex()
        if not cur or (not cur.isValid() and self.selection_model.hasSelection()):
            cur = self.selection_model.selectedIndexes()[0]
        if cur and cur.isValid():
            return cur.sibling(cur.row(), 5).data()

    def current_file(self):
        if False:
            print('Hello World!')
        ' Get the File object for the current files-view item, or the first selection '
        cur_id = self.current_file_id()
        if cur_id:
            return File.get(id=cur_id)
        else:
            return None

    def value_updated(self, item):
        if False:
            return 10
        ' Table cell change event - when tags are updated on a file'
        if item.column() == 2:
            tags_value = item.data(0)
            f = self.current_file()
            if f:
                f.data['tags'] = tags_value
                f.save()

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        app = get_app()
        app.updates.add_listener(self)
        self.model = QStandardItemModel()
        self.model.setColumnCount(6)
        self.model_ids = {}
        self.ignore_updates = False
        self.ignore_image_sequence_paths = []
        self.proxy_model = FileFilterProxyModel(parent=self)
        self.proxy_model.setDynamicSortFilter(True)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy_model.setSortCaseSensitivity(Qt.CaseSensitive)
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setSortLocaleAware(True)
        self.model.itemChanged.connect(self.value_updated)
        self.selection_model = QItemSelectionModel(self.proxy_model)
        app.window.FileUpdated.connect(self.update_file_thumbnail)
        app.window.refreshFilesSignal.connect(functools.partial(self.update_model, clear=False))
        super(QObject, FilesModel).__init__(self, *args)
        if info.MODEL_TEST:
            try:
                from PyQt5.QtTest import QAbstractItemModelTester
                self.model_tests = []
                for m in [self.proxy_model, self.model]:
                    self.model_tests.append(QAbstractItemModelTester(m, QAbstractItemModelTester.FailureReportingMode.Warning))
                log.info('Enabled {} model tests for emoji data'.format(len(self.model_tests)))
            except ImportError:
                pass