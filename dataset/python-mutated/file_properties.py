"""
 @file
 @brief This file loads the File Properties dialog
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
from PyQt5.QtWidgets import QDialog, QFileDialog, QDialogButtonBox, QPushButton
import openshot
from uuid import uuid4
from classes import info, ui_util
from classes.app import get_app
from classes.image_types import get_media_type
from classes.logger import log
from classes.metrics import track_metric_screen
from classes.query import Clip

class FileProperties(QDialog):
    """ File Properties Dialog """
    ui_path = os.path.join(info.PATH, 'windows', 'ui', 'file-properties.ui')

    def __init__(self, file):
        if False:
            return 10
        self.file = file
        QDialog.__init__(self)
        ui_util.load_ui(self, self.ui_path)
        ui_util.init_ui(self)
        app = get_app()
        _ = app._tr
        self.s = app.get_settings()
        track_metric_screen('file-properties-screen')
        self.update_button = QPushButton(_('Update'))
        self.buttonBox.addButton(self.update_button, QDialogButtonBox.AcceptRole)
        self.buttonBox.addButton(QPushButton(_('Cancel')), QDialogButtonBox.RejectRole)
        self.settings_data = self.s.get_all_settings()
        self.channel_layout_choices = []
        self.initialize()

    def initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Init all form elements / textboxes / etc...'
        app = get_app()
        _ = app._tr
        filename = os.path.basename(self.file.data['path'])
        file_extension = os.path.splitext(filename)[1]
        tags = self.file.data.get('tags', '')
        name = self.file.data.get('name', filename)
        self.txtFileName.setText(name)
        self.txtTags.setText(tags)
        self.txtFilePath.setText(self.file.data['path'])
        self.btnBrowse.clicked.connect(self.browsePath)
        self.txtWidth.setValue(self.file.data['width'])
        self.txtHeight.setValue(self.file.data['height'])
        self.txtFrameRateNum.setValue(self.file.data['fps']['num'])
        self.txtFrameRateDen.setValue(self.file.data['fps']['den'])
        self.txtAspectRatioNum.setValue(self.file.data['display_ratio']['num'])
        self.txtAspectRatioDen.setValue(self.file.data['display_ratio']['den'])
        self.txtPixelRatioNum.setValue(self.file.data['pixel_ratio']['num'])
        self.txtPixelRatioDen.setValue(self.file.data['pixel_ratio']['den'])
        if self.file.data['has_audio']:
            self.txtFrameRateNum.setEnabled(False)
            self.txtFrameRateDen.setEnabled(False)
        self.init_start_end_textboxes(self.file.data)
        self.txtVideoFormat.setText(file_extension.replace('.', ''))
        self.txtVideoCodec.setText(self.file.data['vcodec'])
        self.txtAudioCodec.setText(self.file.data['acodec'])
        self.txtSampleRate.setValue(int(self.file.data['sample_rate']))
        self.txtChannels.setValue(int(self.file.data['channels']))
        self.txtVideoBitRate.setValue(int(self.file.data['video_bit_rate']))
        self.txtAudioBitRate.setValue(int(self.file.data['audio_bit_rate']))
        self.txtOutput.setText(json.dumps(self.file.data, sort_keys=True, indent=2))
        selected_channel_layout_index = 0
        current_channel_layout = 0
        if self.file.data['has_audio']:
            current_channel_layout = int(self.file.data['channel_layout'])
        self.channel_layout_choices = []
        layouts = [(0, _('Unknown')), (openshot.LAYOUT_MONO, _('Mono (1 Channel)')), (openshot.LAYOUT_STEREO, _('Stereo (2 Channel)')), (openshot.LAYOUT_SURROUND, _('Surround (3 Channel)')), (openshot.LAYOUT_5POINT1, _('Surround (5.1 Channel)')), (openshot.LAYOUT_7POINT1, _('Surround (7.1 Channel)'))]
        for (channel_layout_index, layout) in enumerate(layouts):
            log.info(layout)
            self.channel_layout_choices.append(layout[0])
            self.cboChannelLayout.addItem(layout[1], layout[0])
            if current_channel_layout == layout[0]:
                selected_channel_layout_index = channel_layout_index
        self.cboChannelLayout.setCurrentIndex(selected_channel_layout_index)
        self.cboInterlaced.clear()
        self.cboInterlaced.addItem(_('Yes'), 'Yes')
        self.cboInterlaced.addItem(_('No'), 'No')
        if self.file.data['interlaced_frame']:
            self.cboInterlaced.setCurrentIndex(0)
        else:
            self.cboInterlaced.setCurrentIndex(1)
        self.toolBox.setCurrentIndex(0)

    def init_start_end_textboxes(self, file_object):
        if False:
            while True:
                i = 10
        'Initialize the start and end textboxes based on a file object'
        fps_float = float(file_object['fps']['num']) / float(file_object['fps']['den'])
        self.txtStartFrame.setMaximum(int(file_object['video_length']))
        if 'start' not in file_object.keys():
            self.txtStartFrame.setValue(1)
        else:
            self.txtStartFrame.setValue(round(float(file_object['start']) * fps_float) + 1)
        self.txtEndFrame.setMaximum(int(file_object['video_length']))
        if 'end' not in file_object.keys():
            self.txtEndFrame.setValue(int(file_object['video_length']))
        else:
            self.txtEndFrame.setValue(round(float(file_object['end']) * fps_float) + 1)

    def verifyPath(self, new_path):
        if False:
            print('Hello World!')
        'If the path has changed, verify that path is valid, and\n        update duration, video_length, media_type, etc...'
        seq_info = get_app().window.files_model.get_image_sequence_details(new_path)
        get_app().window.files_model.ignore_image_sequence_paths = []
        if seq_info:
            new_path = seq_info.get('path')
            self.file.data['media_type'] = 'video'
        clip = openshot.Clip(new_path)
        if clip and clip.info.duration > 0.0:
            self.txtFilePath.setText(new_path)
            self.txtFileName.setText(os.path.basename(new_path))
            self.file.data = json.loads(clip.Reader().Json())
            if not seq_info:
                self.file.data['media_type'] = get_media_type(self.file.data)
            self.init_start_end_textboxes(self.file.data)
        else:
            log.info(f"Given path '{new_path}' was not a valid path... ignoring")

    def browsePath(self):
        if False:
            i = 10
            return i + 15
        app = get_app()
        _ = app._tr
        (starting_folder, filename) = os.path.split(self.file.data['path'])
        new_path = QFileDialog.getOpenFileName(None, _('Locate media file: %s') % filename, starting_folder)[0]
        if new_path:
            self.verifyPath(new_path)
            self.initialize()

    def accept(self):
        if False:
            print('Hello World!')
        new_path = self.txtFilePath.text()
        if new_path and self.file.data.get('path') != new_path:
            self.verifyPath(new_path)
        self.file.data['name'] = self.txtFileName.text()
        self.file.data['tags'] = self.txtTags.text()
        fps_float = self.txtFrameRateNum.value() / self.txtFrameRateDen.value()
        if self.file.data['fps']['num'] != self.txtFrameRateNum.value() or self.file.data['fps']['den'] != self.txtFrameRateDen.value():
            original_fps_float = float(self.file.data['fps']['num']) / float(self.file.data['fps']['den'])
            self.file.data['fps']['num'] = self.txtFrameRateNum.value()
            self.file.data['fps']['den'] = self.txtFrameRateDen.value()
            self.file.data['video_timebase']['num'] = self.txtFrameRateDen.value()
            self.file.data['video_timebase']['den'] = self.txtFrameRateNum.value()
            fps_diff = original_fps_float / fps_float
            self.file.data['duration'] *= fps_diff
            if 'start' in self.file.data:
                self.file.data['start'] *= fps_diff
            if 'end' in self.file.data:
                self.file.data['end'] *= fps_diff
        elif self.txtStartFrame.value() != 1 or self.txtEndFrame.value() != int(self.file.data['video_length']):
            self.file.data['start'] = (self.txtStartFrame.value() - 1) / fps_float
            self.file.data['end'] = (self.txtEndFrame.value() - 1) / fps_float
        tid = str(uuid4())
        get_app().updates.transaction_id = tid
        self.file.save()
        get_app().window.FileUpdated.emit(self.file.id)
        for clip in Clip.filter(file_id=self.file.id):
            clip.data['reader'] = self.file.data
            clip.data['duration'] = self.file.data['duration']
            if clip.data['end'] > clip.data['duration']:
                clip.data['end'] = clip.data['duration']
            clip.save()
            thumbnail_frame = clip.data['start'] * fps_float + 1
            get_app().window.ThumbnailUpdated.emit(clip.id, thumbnail_frame)
        get_app().updates.transaction_id = None
        super(FileProperties, self).accept()

    def reject(self):
        if False:
            i = 10
            return i + 15
        super(FileProperties, self).reject()