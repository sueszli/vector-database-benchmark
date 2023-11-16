import unittest
from binascii import hexlify
from functools import partial
from threading import Event, Thread
from typing import List
from unittest.mock import patch
import requests_mock
from streamlink import Streamlink
from streamlink.stream.hls import HLSStream, HLSStreamWorker as _HLSStreamWorker, HLSStreamWriter as _HLSStreamWriter
from tests.testutils.handshake import Handshake
TIMEOUT_AWAIT_READ = 5
TIMEOUT_AWAIT_READ_ONCE = 5
TIMEOUT_AWAIT_WRITE = 60
TIMEOUT_AWAIT_PLAYLIST_RELOAD = 5
TIMEOUT_AWAIT_PLAYLIST_WAIT = 5
TIMEOUT_AWAIT_CLOSE = 5

class HLSItemBase:
    path = ''

    def url(self, namespace):
        if False:
            return 10
        return 'http://mocked/{namespace}/{path}'.format(namespace=namespace, path=self.path)

class Playlist(HLSItemBase):
    path = 'playlist.m3u8'

    def __init__(self, mediasequence=None, segments=None, end=False, targetduration=0, version=7):
        if False:
            while True:
                i = 10
        self.items = [Tag('EXTM3U'), Tag('EXT-X-VERSION', int(version)), Tag('EXT-X-TARGETDURATION', int(targetduration))]
        if mediasequence is not None:
            self.items.append(Tag('EXT-X-MEDIA-SEQUENCE', int(mediasequence)))
        self.items += segments or []
        if end:
            self.items.append(Tag('EXT-X-ENDLIST'))

    def build(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return '\n'.join([item.build(*args, **kwargs) for item in self.items])

class Tag(HLSItemBase):

    def __init__(self, name, attrs=None):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.attrs = attrs

    @classmethod
    def val_quoted_string(cls, value):
        if False:
            i = 10
            return i + 15
        return '"{0}"'.format(value)

    @classmethod
    def val_hex(cls, value):
        if False:
            while True:
                i = 10
        return '0x{0}'.format(hexlify(value).decode('ascii'))

    def build(self, *args, **kwargs):
        if False:
            return 10
        attrs = None
        if isinstance(self.attrs, dict):
            attrs = ','.join(['{0}={1}'.format(key, value(self, *args, **kwargs) if callable(value) else value) for (key, value) in self.attrs.items()])
        elif self.attrs is not None:
            attrs = str(self.attrs)
        return '#{name}{attrs}'.format(name=self.name, attrs=':{0}'.format(attrs) if attrs else '')

class Segment(HLSItemBase):

    def __init__(self, num, title=None, duration=None, path_relative=True):
        if False:
            i = 10
            return i + 15
        self.num = int(num or 0)
        self.title = str(title or '')
        self.duration = float(duration or 1)
        self.path_relative = bool(path_relative)
        self.content = '[{0}]'.format(self.num).encode('ascii')

    @property
    def path(self):
        if False:
            for i in range(10):
                print('nop')
        return 'segment{0}.ts'.format(self.num)

    def build(self, namespace):
        if False:
            print('Hello World!')
        return '#EXTINF:{duration:.3f},{title}\n{path}'.format(duration=self.duration, title=self.title, path=self.path if self.path_relative else self.url(namespace))

class EventedHLSStreamWorker(_HLSStreamWorker):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.handshake_reload = Handshake()
        self.handshake_wait = Handshake()
        self.time_wait = None

    def reload_playlist(self):
        if False:
            print('Hello World!')
        with self.handshake_reload():
            return super().reload_playlist()

    def wait(self, time):
        if False:
            i = 10
            return i + 15
        self.time_wait = time
        with self.handshake_wait():
            return not self.closed

class EventedHLSStreamWriter(_HLSStreamWriter):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.handshake = Handshake()

    def _queue_put(self, item):
        if False:
            i = 10
            return i + 15
        self._queue.put_nowait(item)

    def _queue_get(self):
        if False:
            for i in range(10):
                print('nop')
        return self._queue.get_nowait()

    @staticmethod
    def _future_result(future):
        if False:
            while True:
                i = 10
        return future.result(timeout=0)

    def write(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        with self.handshake(Exception) as cm:
            if not self.closed:
                super().write(*args, **kwargs)
        if cm.error:
            self.reader.close()

class HLSStreamReadThread(Thread):
    """
    Run the reader on a separate thread, so that each read can be controlled from within the main thread
    """

    def __init__(self, session: Streamlink, stream: HLSStream, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs, daemon=True)
        self.read_once = Event()
        self.handshake = Handshake()
        self.read_all = False
        self.data: List[bytes] = []
        self.session = session
        self.stream = stream
        self.reader = stream.__reader__(stream)

        def _await_read_then_close():
            if False:
                print('Hello World!')
            self.read_once.wait(timeout=TIMEOUT_AWAIT_READ_ONCE)
            return self.writer_close()
        self.writer_close = self.reader.writer.close
        self.reader.writer.close = _await_read_then_close

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while not self.reader.buffer.closed:
            self.read_once.set()
            with self.handshake(OSError) as cm:
                if self.reader.buffer.closed and self.reader.buffer.length == 0:
                    return
                if self.read_all:
                    self.data += list(iter(partial(self.reader.read, -1), b''))
                    return
                self.data.append(self.reader.read(-1))
            if cm.error:
                return

    def reset(self):
        if False:
            while True:
                i = 10
        self.data.clear()

    def close(self):
        if False:
            return 10
        self.reader.close()
        self.read_once.set()
        self.handshake.go()

class TestMixinStreamHLS(unittest.TestCase):
    __stream__ = HLSStream
    __readthread__ = HLSStreamReadThread
    session: Streamlink
    stream: HLSStream
    thread: HLSStreamReadThread

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._patch_http_retry_sleep = patch('streamlink.plugin.api.http_session.time.sleep')
        self.mocker = requests_mock.Mocker()
        self.mocks = {}

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self._patch_http_retry_sleep.start()
        self.mocker.start()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        self.close()
        self.await_close()
        self.mocker.stop()
        self.mocks.clear()
        self._patch_http_retry_sleep.stop()

    def mock(self, method, url, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.mocks[url] = self.mocker.request(method, url, *args, **kwargs)

    def get_mock(self, item):
        if False:
            for i in range(10):
                print('nop')
        return self.mocks[self.url(item)]

    def called(self, item, once=False):
        if False:
            i = 10
            return i + 15
        mock = self.get_mock(item)
        return mock.call_count == 1 if once else mock.called

    def url(self, item):
        if False:
            return 10
        return item.url(self.id())

    @staticmethod
    def content(segments, prop='content', cond=None):
        if False:
            return 10
        if isinstance(segments, dict):
            segments = segments.values()
        return b''.join([getattr(segment, prop) for segment in segments if cond is None or cond(segment)])

    def await_close(self, timeout=TIMEOUT_AWAIT_CLOSE):
        if False:
            for i in range(10):
                print('nop')
        thread = self.thread
        thread.reader.writer.join(timeout)
        thread.reader.worker.join(timeout)
        thread.join(timeout)
        assert self.thread.reader.closed, 'Stream reader is closed'

    def await_playlist_reload(self, timeout=TIMEOUT_AWAIT_PLAYLIST_RELOAD) -> None:
        if False:
            while True:
                i = 10
        worker: EventedHLSStreamWorker = self.thread.reader.worker
        assert worker.is_alive()
        assert worker.handshake_reload.step(timeout)

    def await_playlist_wait(self, timeout=TIMEOUT_AWAIT_PLAYLIST_WAIT) -> None:
        if False:
            return 10
        worker: EventedHLSStreamWorker = self.thread.reader.worker
        assert worker.is_alive()
        assert worker.handshake_wait.step(timeout)

    def await_write(self, write_calls=1, timeout=TIMEOUT_AWAIT_WRITE) -> None:
        if False:
            for i in range(10):
                print('nop')
        writer: EventedHLSStreamWriter = self.thread.reader.writer
        assert writer.is_alive()
        for _ in range(write_calls):
            assert writer.handshake.step(timeout)

    def await_read(self, read_all=False, timeout=TIMEOUT_AWAIT_READ):
        if False:
            while True:
                i = 10
        thread = self.thread
        thread.read_all = read_all
        assert thread.is_alive()
        assert thread.handshake.step(timeout)
        data = b''.join(thread.data)
        thread.reset()
        return data

    def get_session(self, options=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return Streamlink(options)

    def subject(self, playlists, options=None, streamoptions=None, threadoptions=None, start=True, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        segments_all = [item for playlist in playlists for item in playlist.items if isinstance(item, Segment)]
        segments = {segment.num: segment for segment in segments_all}
        self.mock('GET', self.url(playlists[0]), [{'text': pl.build(self.id())} for pl in playlists])
        for segment in segments.values():
            self.mock('GET', self.url(segment), content=segment.content)
        self.session = self.get_session(options, *args, **kwargs)
        self.stream = self.__stream__(self.session, self.url(playlists[0]), **streamoptions or {})
        self.thread = self.__readthread__(self.session, self.stream, name=f'ReadThread-{self.id()}', **threadoptions or {})
        if start:
            self.start()
        return segments

    def start(self):
        if False:
            return 10
        self.thread.reader.open()
        self.thread.start()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        thread = self.thread
        thread.reader.close()
        if isinstance(thread.reader.writer, EventedHLSStreamWriter):
            thread.reader.writer.handshake.go()
        thread.handshake.go()