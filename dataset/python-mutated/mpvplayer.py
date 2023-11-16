import locale
import logging
from mpv import MPV, MpvEventID, MpvEventEndFile, _mpv_set_property_string, _mpv_set_option_string, _mpv_client_api_version, ErrorCode
from feeluown.utils.dispatch import Signal
from feeluown.media import Media, VideoAudioManifest
from .base_player import AbstractPlayer, State
from .metadata import MetadataFields, Metadata
logger = logging.getLogger(__name__)

class MpvPlayer(AbstractPlayer):
    """

    player will always play playlist current song. player will listening to
    playlist ``song_changed`` signal and change the current playback.

    todo: make me singleton
    """

    def __init__(self, _=None, audio_device=b'auto', winid=None, **kwargs):
        if False:
            return 10
        '\n        :param _: keep this arg to keep backward compatibility\n        '
        super().__init__(**kwargs)
        locale.setlocale(locale.LC_NUMERIC, 'C')
        mpvkwargs = {}
        if winid is not None:
            mpvkwargs['wid'] = winid
        self._version = _mpv_client_api_version()
        if self._version < (1, 107):
            mpvkwargs['vo'] = 'opengl-cb'
            self.use_opengl_cb = True
        else:
            self.use_opengl_cb = False
        self._mpv = MPV(ytdl=False, input_default_bindings=True, input_vo_keyboard=True, **mpvkwargs)
        _mpv_set_property_string(self._mpv.handle, b'audio-device', audio_device)
        _mpv_set_option_string(self._mpv.handle, b'user-agent', b'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
        self.video_format_changed = Signal()
        self._mpv.observe_property('time-pos', lambda name, position: self._on_position_changed(position))
        self._mpv.observe_property('duration', lambda name, duration: self._on_duration_changed(duration))
        self._mpv.observe_property('video-format', lambda name, vformat: self._on_video_format_changed(vformat))
        self._mpv._event_callbacks.append(self._on_event)
        logger.debug('Player initialize finished.')

    def shutdown(self):
        if False:
            return 10
        if self._mpv.handle is not None:
            self._mpv.terminate()

    def play(self, media, video=True, metadata=None):
        if False:
            while True:
                i = 10
        if video is False:
            _mpv_set_property_string(self._mpv.handle, b'vid', b'no')
        else:
            _mpv_set_property_string(self._mpv.handle, b'vid', b'auto')
        self.media_about_to_changed.emit(self._current_media, media)
        if media is None:
            self._stop_mpv()
        else:
            logger.debug("Player will play: '%s'", media)
            if isinstance(media, Media):
                media = media
            else:
                media = Media(media)
            self._set_http_headers(media.http_headers)
            self._set_http_proxy(media.http_proxy)
            self._stop_mpv()
            if media.manifest is None:
                url = media.url
                self._mpv.play(url)
            elif isinstance(media.manifest, VideoAudioManifest):
                video_url = media.manifest.video_url
                audio_url = media.manifest.audio_url

                def add_audio():
                    if False:
                        while True:
                            i = 10
                    try:
                        if self.current_media is media:
                            self._mpv.audio_add(audio_url)
                            self.resume()
                    finally:
                        self.media_loaded.disconnect(add_audio)
                if video is True:
                    self._mpv.play(video_url)
                    self.media_loaded.connect(add_audio, weak=False)
                else:
                    self._mpv.play(audio_url)
            else:
                assert False, 'Unknown manifest'
        self._current_media = media
        self.media_changed.emit(media)
        if metadata is None:
            self._current_metadata = {}
        else:
            metadata['__setby__'] = 'manual'
            self._current_metadata = metadata
        self.metadata_changed.emit(self.current_metadata)

    def set_play_range(self, start=None, end=None):
        if False:
            i = 10
            return i + 15
        if self._version >= (1, 28):
            (start_default, end_default) = ('none', 'none')
        else:
            (start_default, end_default) = ('0%', '100%')
        start_str = str(start) if start is not None else start_default
        end_str = str(end) if end is not None else end_default
        _mpv_set_option_string(self._mpv.handle, b'start', bytes(start_str, 'utf-8'))
        if start is not None:
            self.seeked.emit(start)
        _mpv_set_option_string(self._mpv.handle, b'end', bytes(end_str, 'utf-8'))

    def resume(self):
        if False:
            while True:
                i = 10
        self._mpv.pause = False
        self.state = State.playing

    def pause(self):
        if False:
            print('Hello World!')
        self._mpv.pause = True
        self.state = State.paused

    def toggle(self):
        if False:
            return 10
        self._mpv.pause = not self._mpv.pause
        if self._mpv.pause:
            self.state = State.paused
        else:
            self.state = State.playing

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        self._mpv.pause = True
        self.state = State.stopped
        self.play(None)
        logger.debug('Player stopped.')

    @property
    def position(self):
        if False:
            return 10
        return self._position

    @position.setter
    def position(self, position):
        if False:
            while True:
                i = 10
        if self._current_media:
            self._mpv.seek(position, reference='absolute')
            self._position = position
            self.seeked.emit(position)
        else:
            logger.warn("can't set position when current media is empty")

    @AbstractPlayer.volume.setter
    def volume(self, value):
        if False:
            print('Hello World!')
        super(MpvPlayer, MpvPlayer).volume.__set__(self, value)
        self._mpv.volume = self.volume

    @property
    def video_format(self):
        if False:
            i = 10
            return i + 15
        return self._video_format

    @video_format.setter
    def video_format(self, vformat):
        if False:
            print('Hello World!')
        self._video_format = vformat
        self.video_format_changed.emit(vformat)

    def _stop_mpv(self):
        if False:
            i = 10
            return i + 15
        self._mpv.play('')
        self._mpv.playlist_clear()

    def _on_position_changed(self, position):
        if False:
            return 10
        self._position = max(0, position or 0)
        self.position_changed.emit(position)

    def _on_duration_changed(self, duration):
        if False:
            i = 10
            return i + 15
        'listening to mpv duration change event'
        logger.debug('Player receive duration changed signal')
        self.duration = duration

    def _on_video_format_changed(self, vformat):
        if False:
            i = 10
            return i + 15
        self.video_format = vformat

    def _on_event(self, event):
        if False:
            print('Hello World!')
        event_id = event['event_id']
        if event_id == MpvEventID.END_FILE:
            reason = event['event']['reason']
            logger.debug('Current song finished. reason: %d' % reason)
            if self.state != State.stopped and reason != MpvEventEndFile.ABORTED:
                self.media_finished.emit()
                if reason == MpvEventEndFile.ERROR and event['event']['error'] == ErrorCode.LOADING_FAILED:
                    self.media_loading_failed.emit()
        elif event_id == MpvEventID.FILE_LOADED:
            self.media_loaded.emit()
        elif event_id == MpvEventID.METADATA_UPDATE:
            metadata = dict(self._mpv.metadata or {})
            logger.debug('metadata updated to %s', metadata)
            if self._current_metadata.get('__setby__') != 'manual':
                self._current_metadata['__setby__'] = 'automatic'
                mapping = Metadata({MetadataFields.title: 'title', MetadataFields.album: 'album', MetadataFields.artists: 'artist'})
                for (src, tar) in mapping.items():
                    if tar in metadata:
                        value = metadata[tar]
                        if src is MetadataFields.artists:
                            value = [value]
                        self._current_metadata[src] = value
                self.metadata_changed.emit(self.current_metadata)

    def _set_http_headers(self, http_headers):
        if False:
            while True:
                i = 10
        if http_headers:
            headers = []
            for (key, value) in http_headers.items():
                headers.append('{}: {}'.format(key, value))
            headers_text = ','.join(headers)
            headers_bytes = bytes(headers_text, 'utf-8')
            logger.info('play media with headers: %s', headers_text)
            _mpv_set_option_string(self._mpv.handle, b'http-header-fields', headers_bytes)
        else:
            _mpv_set_option_string(self._mpv.handle, b'http-header-fields', b'')

    def _set_http_proxy(self, http_proxy):
        if False:
            print('Hello World!')
        _mpv_set_option_string(self._mpv.handle, b'http-proxy', bytes(http_proxy, 'utf-8'))

    def __log_handler(self, loglevel, component, message):
        if False:
            for i in range(10):
                print('nop')
        print('[{}] {}: {}'.format(loglevel, component, message))