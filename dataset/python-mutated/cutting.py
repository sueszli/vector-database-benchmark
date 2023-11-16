"""
 @file
 @brief This file loads the clip cutting interface (quickly cut up a clip into smaller clips)
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
import functools
import json
from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtWidgets import QDialog, QMessageBox, QSizePolicy
import openshot
from classes import info, ui_util, time_parts
from classes.app import get_app
from classes.logger import log
from classes.metrics import track_metric_screen
from windows.preview_thread import PreviewParent
from windows.video_widget import VideoWidget

class Cutting(QDialog):
    """ Cutting Dialog """
    ui_path = os.path.join(info.PATH, 'windows', 'ui', 'cutting.ui')
    previewFrameSignal = pyqtSignal(int)
    refreshFrameSignal = pyqtSignal()
    LoadFileSignal = pyqtSignal(str)
    PlaySignal = pyqtSignal()
    PauseSignal = pyqtSignal()
    SeekSignal = pyqtSignal(int)
    SpeedSignal = pyqtSignal(float)
    StopSignal = pyqtSignal()

    def __init__(self, file=None, preview=False):
        if False:
            while True:
                i = 10
        _ = get_app()._tr
        QDialog.__init__(self)
        ui_util.load_ui(self, self.ui_path)
        ui_util.init_ui(self)
        track_metric_screen('cutting-screen')
        if preview:
            self.lblInstructions.setVisible(False)
            self.widgetControls.setVisible(False)
            self.setWindowTitle(_('Preview'))
        self.start_frame = 1
        self.start_image = None
        self.end_frame = 1
        self.end_image = None
        self.file = file
        self.file_path = file.absolute_path()
        self.video_length = int(file.data['video_length'])
        self.fps_num = int(file.data['fps']['num'])
        self.fps_den = int(file.data['fps']['den'])
        self.fps = float(self.fps_num) / float(self.fps_den)
        self.width = int(file.data['width'])
        self.height = int(file.data['height'])
        self.sample_rate = int(get_app().project.get('sample_rate'))
        self.channels = int(file.data['channels'])
        self.channel_layout = int(file.data['channel_layout'])
        log.info(self.file_path)
        self.videoPreview = VideoWidget()
        self.videoPreview.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.verticalLayout.insertWidget(0, self.videoPreview)
        viewport_rect = self.videoPreview.centeredViewport(self.videoPreview.width(), self.videoPreview.height())
        self.r = openshot.Timeline(viewport_rect.width(), viewport_rect.height(), openshot.Fraction(self.fps_num, self.fps_den), self.sample_rate, self.channels, self.channel_layout)
        self.r.info.channel_layout = self.channel_layout
        self.r.SetMaxSize(viewport_rect.width(), viewport_rect.height())
        try:
            self.clip = openshot.Clip(self.file_path)
            self.clip.SetJson(json.dumps({'reader': file.data}))
            if not self.clip.Reader().info.has_video and self.clip.Reader().info.has_audio:
                self.clip.Waveform(True)
            self.r.info.has_audio = self.clip.Reader().info.has_audio
            self.r.info.video_length = self.video_length
            if preview:
                self.clip.display = openshot.FRAME_DISPLAY_CLIP
            self.r.AddClip(self.clip)
        except Exception:
            log.error('Failed to load media file into preview player: %s', self.file_path)
            return
        self.r.Open()
        self.initialized = False
        self.transforming_clip = False
        self.preview_parent = PreviewParent()
        self.preview_parent.Init(self, self.r, self.videoPreview, self.video_length)
        self.preview_thread = self.preview_parent.worker
        self.sliderIgnoreSignal = False
        self.sliderVideo.setMinimum(1)
        self.sliderVideo.setMaximum(self.video_length)
        self.sliderVideo.setSingleStep(1)
        self.sliderVideo.setSingleStep(1)
        self.sliderVideo.setPageStep(24)
        start_frame = 1
        if 'start' in self.file.data:
            start_frame = float(self.file.data['start']) * self.fps + 1
        QTimer.singleShot(500, functools.partial(self.sliderVideo.setValue, start_frame + 1))
        QTimer.singleShot(600, functools.partial(self.sliderVideo.setValue, start_frame))
        self.actionPlay.triggered.connect(self.actionPlay_Triggered)
        self.btnPlay.clicked.connect(self.btnPlay_clicked)
        self.sliderVideo.valueChanged.connect(self.sliderVideo_valueChanged)
        self.btnStart.clicked.connect(self.btnStart_clicked)
        self.btnEnd.clicked.connect(self.btnEnd_clicked)
        self.btnClear.clicked.connect(self.btnClear_clicked)
        self.btnAddClip.clicked.connect(self.btnAddClip_clicked)
        self.initialized = True

    def actionPlay_Triggered(self):
        if False:
            while True:
                i = 10
        self.btnPlay.click()

    def movePlayhead(self, frame_number):
        if False:
            i = 10
            return i + 15
        'Update the playhead position'
        self.sliderIgnoreSignal = True
        self.sliderVideo.setValue(frame_number)
        self.sliderIgnoreSignal = False
        seconds = (frame_number - 1) / self.fps
        time_text = time_parts.secondsToTime(seconds, self.fps_num, self.fps_den)
        timestamp = '%s:%s:%s:%s' % (time_text['hour'], time_text['min'], time_text['sec'], time_text['frame'])
        self.lblVideoTime.setText(timestamp)

    def btnPlay_clicked(self, force=None):
        if False:
            while True:
                i = 10
        log.info('btnPlay_clicked')
        if force == 'pause':
            self.btnPlay.setChecked(False)
        elif force == 'play':
            self.btnPlay.setChecked(True)
        if self.btnPlay.isChecked():
            log.info('play (icon to pause)')
            ui_util.setup_icon(self, self.btnPlay, 'actionPlay', 'media-playback-pause')
            self.preview_thread.Play()
        else:
            log.info('pause (icon to play)')
            ui_util.setup_icon(self, self.btnPlay, 'actionPlay', 'media-playback-start')
            self.preview_thread.Pause()
        self.sliderVideo.setFocus()

    def sliderVideo_valueChanged(self, new_frame):
        if False:
            return 10
        if self.preview_thread and (not self.sliderIgnoreSignal):
            log.info('sliderVideo_valueChanged')
            self.btnPlay_clicked(force='pause')
            self.preview_thread.previewFrame(new_frame)

    def btnStart_clicked(self):
        if False:
            while True:
                i = 10
        'Start of clip button was clicked'
        _ = get_app()._tr
        self.btnPlay_clicked(force='pause')
        current_frame = self.sliderVideo.value()
        if self.btnEnd.isEnabled() and current_frame >= self.end_frame:
            msg = QMessageBox()
            msg.setText(_("Please choose valid 'start' and 'end' values for your clip."))
            msg.exec_()
            return
        self.start_frame = current_frame
        self.start_image = os.path.join(info.USER_PATH, 'thumbnail', '%s.png' % self.start_frame)
        self.r.GetFrame(self.start_frame).Thumbnail(self.start_image, 160, 90, '', '', '#000000', True, 'png', 85)
        self.btnStart.setStyleSheet('background-image: url(%s);' % self.start_image.replace('\\', '/'))
        self.btnEnd.setEnabled(True)
        self.btnClear.setEnabled(True)
        self.sliderVideo.setFocus()
        log.info('btnStart_clicked, current frame: %s' % self.start_frame)

    def btnEnd_clicked(self):
        if False:
            print('Hello World!')
        'End of clip button was clicked'
        _ = get_app()._tr
        self.btnPlay_clicked(force='pause')
        current_frame = self.sliderVideo.value()
        if current_frame <= self.start_frame:
            msg = QMessageBox()
            msg.setText(_("Please choose valid 'start' and 'end' values for your clip."))
            msg.exec_()
            return
        self.end_frame = current_frame
        self.end_image = os.path.join(info.USER_PATH, 'thumbnail', '%s.png' % self.end_frame)
        self.r.GetFrame(self.end_frame).Thumbnail(self.end_image, 160, 90, '', '', '#000000', True, 'png', 85)
        self.btnEnd.setStyleSheet('background-image: url(%s);' % self.end_image.replace('\\', '/'))
        self.btnAddClip.setEnabled(True)
        self.sliderVideo.setFocus()
        log.info('btnEnd_clicked, current frame: %s' % self.end_frame)

    def btnClear_clicked(self):
        if False:
            while True:
                i = 10
        'Clear the current clip and reset the form'
        log.info('btnClear_clicked')
        self.clearForm()

    def clearForm(self):
        if False:
            while True:
                i = 10
        'Clear all form controls'
        self.start_frame = 1
        self.end_frame = 1
        self.start_image = ''
        self.end_image = ''
        self.btnStart.setStyleSheet('background-image: None;')
        self.btnEnd.setStyleSheet('background-image: None;')
        self.txtName.setText('')
        self.btnEnd.setEnabled(False)
        self.btnAddClip.setEnabled(False)
        self.btnClear.setEnabled(False)

    def btnAddClip_clicked(self):
        if False:
            i = 10
            return i + 15
        'Add the selected clip to the project'
        log.info('btnAddClip_clicked')
        if 'name' in self.file.data:
            self.file.data.pop('name')
        self.file.id = None
        self.file.key = None
        self.file.type = 'insert'
        self.file.data['start'] = (self.start_frame - 1) / self.fps
        self.file.data['end'] = (self.end_frame - 1) / self.fps
        if self.txtName.text():
            self.file.data['name'] = self.txtName.text()
        self.file.save()
        self.clearForm()

    def accept(self):
        if False:
            i = 10
            return i + 15
        ' Ok button clicked '
        log.info('accept')

    def close(self):
        if False:
            while True:
                i = 10
        ' Actually close window and accept dialog '
        log.info('close')

    def closeEvent(self, event):
        if False:
            i = 10
            return i + 15
        log.debug('closeEvent')
        get_app().updates.disconnect_listener(self.videoPreview)
        if self.videoPreview:
            self.videoPreview.deleteLater()
            self.videoPreview = None
        self.preview_parent.Stop()
        self.r.Close()
        self.clip.Close()
        self.r.ClearAllCache()

    def reject(self):
        if False:
            print('Hello World!')
        log.info('reject')