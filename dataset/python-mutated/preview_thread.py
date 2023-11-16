"""
 @file
 @brief This file contains the preview thread, used for displaying previews of the timeline
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
import time
import sip
import math
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSlot, pyqtSignal, QCoreApplication
from PyQt5.QtWidgets import QMessageBox
import openshot
from classes.app import get_app
from classes.logger import log
from classes.updates import UpdateInterface

class PreviewParent(QObject, UpdateInterface):
    """ Class which communicates with the PlayerWorker Class (running on a separate thread) """

    def changed(self, action):
        if False:
            print('Hello World!')
        ' This method is invoked by the UpdateManager each time a change happens (i.e UpdateInterface) '
        if len(action.key) >= 1 and action.key[0].lower() in ['files', 'history', 'markers', 'layers', 'scale', 'profile', 'sample_rate']:
            return
        try:
            self.timeline_max_length = self.timeline.GetMaxFrame()
            log.debug(f'Max timeline length/frames detected: {self.timeline_max_length}')
        except Exception as e:
            log.info('Error calculating max timeline length on PreviewParent: %s. %s' % (e, action.json(is_array=True)))

    def onPositionChanged(self, current_frame):
        if False:
            print('Hello World!')
        self.parent.movePlayhead(current_frame)
        if self.worker.player.Mode() == openshot.PLAYBACK_PLAY:
            if self.worker.player.Speed() > 0.0 and current_frame >= self.timeline_max_length:
                self.parent.PauseSignal.emit()
                self.worker.Seek(self.timeline_max_length)
            if self.worker.player.Speed() < 0.0 and current_frame <= 1:
                self.parent.PauseSignal.emit()
                self.worker.Seek(1)

    def onModeChanged(self, current_mode):
        if False:
            return 10
        log.debug('Playback mode changed to %s', current_mode)
        try:
            if current_mode is openshot.PLAYBACK_PLAY:
                self.parent.SetPlayheadFollow(False)
            else:
                self.parent.SetPlayheadFollow(True)
        except AttributeError:
            pass

    def onError(self, error):
        if False:
            return 10
        _ = get_app()._tr
        QMessageBox.warning(self.parent, _('Audio Error'), _('Please fix the following error and restart OpenShot\n%s') % error)

    def Stop(self):
        if False:
            for i in range(10):
                print('nop')
        'Disconnect preview parent from update manager and stop worker thread'
        get_app().updates.disconnect_listener(self)
        self.worker.Stop()
        self.worker.kill()
        self.background.exit()
        self.background.wait(5000)

    @pyqtSlot(object, object)
    def Init(self, parent, timeline, video_widget, max_length=1):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.timeline = timeline
        self.timeline_max_length = max_length
        self.background = QThread(self)
        self.worker = PlayerWorker()
        self.worker.Init(parent, timeline, video_widget)
        self.worker.position_changed.connect(self.onPositionChanged)
        self.worker.mode_changed.connect(self.onModeChanged)
        self.background.started.connect(self.worker.Start)
        self.worker.finished.connect(self.background.quit)
        self.worker.error_found.connect(self.onError)
        self.parent.previewFrameSignal.connect(self.worker.previewFrame)
        self.parent.refreshFrameSignal.connect(self.worker.refreshFrame)
        self.parent.LoadFileSignal.connect(self.worker.LoadFile)
        self.parent.PlaySignal.connect(self.worker.Play)
        self.parent.PauseSignal.connect(self.worker.Pause)
        self.parent.SeekSignal.connect(self.worker.Seek)
        self.parent.SpeedSignal.connect(self.worker.Speed)
        self.parent.StopSignal.connect(self.worker.Stop)
        self.worker.moveToThread(self.background)
        self.background.start()
        get_app().updates.add_listener(self)

class PlayerWorker(QObject):
    """ QT Player Worker Object (to preview video on a separate thread) """
    position_changed = pyqtSignal(int)
    mode_changed = pyqtSignal(object)
    error_found = pyqtSignal(object)
    finished = pyqtSignal()

    @pyqtSlot(object, object)
    def Init(self, parent, timeline, videoPreview):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.timeline = timeline
        self.videoPreview = videoPreview
        self.clip_path = None
        self.clip_reader = None
        self.original_speed = 0
        self.original_position = 0
        self.previous_clips = []
        self.previous_clip_readers = []
        self.is_running = True
        self.number = None
        self.current_frame = None
        self.current_mode = None
        self.player = openshot.QtPlayer()

    def CheckAudioDevice(self):
        if False:
            while True:
                i = 10
        'Check if any audio devices initialization errors, default sample rate, and current open audio device'
        audio_error = self.player.GetError()
        if audio_error:
            log.warning('Audio initialization error: %s', audio_error)
            self.error_found.emit(audio_error)
        detected_sample_rate = float(self.player.GetDefaultSampleRate())
        if detected_sample_rate and (not math.isnan(detected_sample_rate)) and (detected_sample_rate > 0.0):
            detected_sample_rate_int = round(detected_sample_rate)
            s = get_app().get_settings()
            settings_sample_rate = int(s.get('default-samplerate') or 48000)
            if detected_sample_rate_int != settings_sample_rate:
                log.warning("Your sample rate (%d) does not match OpenShot (%d). Adjusting your 'Preferences->Preview->Default Sample Rate to match your system rate: %d." % (detected_sample_rate_int, settings_sample_rate, detected_sample_rate_int))
                s.set('default-samplerate', detected_sample_rate_int)
                get_app().updates.update(['sample_rate'], detected_sample_rate_int)
        if type(s.get('default-samplerate')) == float:
            s.set('default-samplerate', detected_sample_rate_int)
        if type(get_app().project.get('sample_rate')) == float:
            get_app().updates.update(['sample_rate'], round(get_app().project.get('sample_rate')))
        active_audio_device = self.player.GetCurrentAudioDevice()
        audio_device_value = f'{active_audio_device.get_name()}||{active_audio_device.get_type()}'
        if s.get('playback-audio-device') != audio_device_value:
            log.warning("Your active audio device (%s) does not match OpenShot (%s). Adjusting your 'Preferences->Playback->Audio Device' to match your active audio device: %s" % (audio_device_value, s.get('playback-audio-device'), audio_device_value))
            s.set('playback-audio-device', audio_device_value)
            lib_settings = openshot.Settings.Instance()
            lib_settings.PLAYBACK_AUDIO_DEVICE_NAME = active_audio_device.get_name()
            lib_settings.PLAYBACK_AUDIO_DEVICE_TYPE = active_audio_device.get_type()

    @pyqtSlot()
    def Start(self):
        if False:
            print('Hello World!')
        ' This method starts the video player '
        log.info('QThread Start Method Invoked')
        self.initPlayer()
        self.player.Reader(self.timeline)
        self.player.Play()
        self.player.Pause()
        QTimer.singleShot(1000, self.CheckAudioDevice)
        while self.is_running:
            if self.current_frame != self.player.Position():
                self.current_frame = self.player.Position()
                if not self.clip_path:
                    self.position_changed.emit(self.current_frame)
                    QCoreApplication.processEvents()
            if self.player.Mode() != self.current_mode:
                self.current_mode = self.player.Mode()
                self.mode_changed.emit(self.current_mode)
            time.sleep(0.01)
            QCoreApplication.processEvents()
        self.finished.emit()
        log.debug('exiting playback thread')

    @pyqtSlot()
    def initPlayer(self):
        if False:
            while True:
                i = 10
        log.debug('initPlayer')
        self.renderer_address = self.player.GetRendererQObject()
        self.player.SetQWidget(sip.unwrapinstance(self.videoPreview))
        self.renderer = sip.wrapinstance(self.renderer_address, QObject)
        self.videoPreview.connectSignals(self.renderer)

    def kill(self):
        if False:
            return 10
        ' Kill this thread '
        self.is_running = False

    def previewFrame(self, number):
        if False:
            print('Hello World!')
        ' Preview a certain frame '
        self.Seek(number)
        log.debug('previewFrame: %s, player Position(): %s', number, self.player.Position())

    def refreshFrame(self):
        if False:
            i = 10
            return i + 15
        ' Refresh a certain frame '
        log.debug('refreshFrame')
        self.parent.LoadFileSignal.emit('')
        self.Seek(self.player.Position())
        log.debug('player Position(): %s', self.player.Position())

    def LoadFile(self, path=None):
        if False:
            i = 10
            return i + 15
        ' Load a media file into the video player '
        if path == self.clip_path or (not path and (not self.clip_path)):
            return
        log.info('LoadFile %s' % path)
        seek_position = 1
        if path and (not self.clip_path):
            self.original_position = self.player.Position()
        if not path:
            log.debug('Set timeline reader again in player: %s' % self.timeline)
            self.player.Reader(self.timeline)
            self.clip_reader = None
            self.clip_path = None
            seek_position = self.original_position
        else:
            project = get_app().project
            fps = project.get('fps')
            width = int(project.get('width'))
            height = int(project.get('height'))
            sample_rate = int(project.get('sample_rate'))
            channels = int(project.get('channels'))
            channel_layout = int(project.get('channel_layout'))
            self.clip_reader = openshot.Timeline(width, height, openshot.Fraction(fps['num'], fps['den']), sample_rate, channels, channel_layout)
            self.clip_reader.info.channel_layout = channel_layout
            self.clip_reader.info.has_audio = True
            self.clip_reader.info.has_video = True
            self.clip_reader.info.video_length = 999999
            self.clip_reader.info.duration = 999999
            self.clip_reader.info.sample_rate = sample_rate
            self.clip_reader.info.channels = channels
            try:
                new_clip = openshot.Clip(path)
                self.clip_reader.AddClip(new_clip)
            except:
                log.error('Failed to load media file into video player: %s' % path)
                return
            self.clip_path = path
            self.previous_clips.append(new_clip)
            self.previous_clip_readers.append(self.clip_reader)
            self.clip_reader.Open()
            self.player.Reader(self.clip_reader)
        while len(self.previous_clip_readers) > 3:
            log.debug('Removing old clips from preview: %s' % self.previous_clip_readers[0])
            previous_clip = self.previous_clips.pop(0)
            previous_clip.Close()
            previous_reader = self.previous_clip_readers.pop(0)
            previous_reader.Close()
        self.Seek(seek_position)

    def Play(self):
        if False:
            i = 10
            return i + 15
        ' Start playing the video player '
        if self.parent.initialized:
            self.player.Play()

    def Pause(self):
        if False:
            while True:
                i = 10
        ' Pause the video player '
        if self.parent.initialized:
            self.player.Pause()

    def Stop(self):
        if False:
            return 10
        ' Stop the video player and terminate the playback threads '
        if self.parent.initialized:
            self.player.Stop()

    def Seek(self, number):
        if False:
            return 10
        ' Seek to a specific frame '
        if self.parent.initialized:
            self.player.Seek(number)

    def Speed(self, new_speed):
        if False:
            i = 10
            return i + 15
        ' Set the speed of the video player '
        if self.parent.initialized and self.player.Speed() != new_speed:
            self.player.Speed(new_speed)