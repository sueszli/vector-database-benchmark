import asyncio
import errno
import os
import tempfile
import time
import wave
from unittest import TestCase
from unittest.mock import patch
import av
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError, VideoStreamTrack
from .codecs import CodecTestCase
from .utils import asynctest

class VideoStreamTrackUhd(VideoStreamTrack):

    async def recv(self):
        (pts, time_base) = await self.next_timestamp()
        frame = av.VideoFrame(width=3840, height=2160)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = pts
        frame.time_base = time_base
        return frame

class MediaTestCase(CodecTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.directory.cleanup()

    def create_audio_file(self, name, channels=1, sample_rate=8000, sample_width=2):
        if False:
            print('Hello World!')
        path = self.temporary_path(name)
        writer = wave.open(path, 'wb')
        writer.setnchannels(channels)
        writer.setframerate(sample_rate)
        writer.setsampwidth(sample_width)
        writer.writeframes(b'\x00' * sample_rate * sample_width * channels)
        writer.close()
        return path

    def create_audio_and_video_file(self, name, width=640, height=480, video_rate=30, duration=1):
        if False:
            i = 10
            return i + 15
        path = self.temporary_path(name)
        audio_pts = 0
        audio_rate = 48000
        audio_samples = audio_rate // video_rate
        container = av.open(path, 'w')
        audio_stream = container.add_stream('libopus', rate=audio_rate)
        video_stream = container.add_stream('h264', rate=video_rate)
        for video_frame in self.create_video_frames(width=width, height=height, count=duration * video_rate):
            audio_frame = self.create_audio_frame(samples=audio_samples, pts=audio_pts, sample_rate=audio_rate)
            audio_pts += audio_samples
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            for packet in video_stream.encode(video_frame):
                container.mux(packet)
        for packet in audio_stream.encode(None):
            container.mux(packet)
        for packet in video_stream.encode(None):
            container.mux(packet)
        container.close()
        return path

    def create_video_file(self, name, width=640, height=480, rate=30, duration=1):
        if False:
            i = 10
            return i + 15
        path = self.temporary_path(name)
        container = av.open(path, 'w')
        if name.endswith('.png'):
            stream = container.add_stream('png', rate=rate)
            stream.pix_fmt = 'rgb24'
        elif name.endswith('.ts'):
            stream = container.add_stream('h264', rate=rate)
        else:
            assert name.endswith('.mp4')
            stream = container.add_stream('mpeg4', rate=rate)
        for frame in self.create_video_frames(width=width, height=height, count=duration * rate):
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()
        return path

    def temporary_path(self, name):
        if False:
            i = 10
            return i + 15
        return os.path.join(self.directory.name, name)

class MediaBlackholeTest(TestCase):

    @asynctest
    async def test_audio(self):
        recorder = MediaBlackhole()
        recorder.addTrack(AudioStreamTrack())
        await recorder.start()
        await asyncio.sleep(1)
        await recorder.stop()

    @asynctest
    async def test_audio_ended(self):
        track = AudioStreamTrack()
        recorder = MediaBlackhole()
        recorder.addTrack(track)
        await recorder.start()
        await asyncio.sleep(1)
        track.stop()
        await asyncio.sleep(1)
        await recorder.stop()

    @asynctest
    async def test_audio_and_video(self):
        recorder = MediaBlackhole()
        recorder.addTrack(AudioStreamTrack())
        recorder.addTrack(VideoStreamTrack())
        await recorder.start()
        await asyncio.sleep(2)
        await recorder.stop()

    @asynctest
    async def test_video(self):
        recorder = MediaBlackhole()
        recorder.addTrack(VideoStreamTrack())
        await recorder.start()
        await asyncio.sleep(2)
        await recorder.stop()

    @asynctest
    async def test_video_ended(self):
        track = VideoStreamTrack()
        recorder = MediaBlackhole()
        recorder.addTrack(track)
        await recorder.start()
        await asyncio.sleep(1)
        track.stop()
        await asyncio.sleep(1)
        await recorder.stop()

class MediaRelayTest(MediaTestCase):

    @asynctest
    async def test_audio_stop_consumer(self):
        source = AudioStreamTrack()
        relay = MediaRelay()
        proxy1 = relay.subscribe(source)
        proxy2 = relay.subscribe(source)
        samples_per_frame = 160
        for pts in range(0, 2 * samples_per_frame, samples_per_frame):
            (frame1, frame2) = await asyncio.gather(proxy1.recv(), proxy2.recv())
            self.assertEqual(frame1.format.name, 's16')
            self.assertEqual(frame1.layout.name, 'mono')
            self.assertEqual(frame1.pts, pts)
            self.assertEqual(frame1.samples, samples_per_frame)
            self.assertEqual(frame2.format.name, 's16')
            self.assertEqual(frame2.layout.name, 'mono')
            self.assertEqual(frame2.pts, pts)
            self.assertEqual(frame2.samples, samples_per_frame)
        proxy1.stop()
        for i in range(2):
            (exc1, frame2) = await asyncio.gather(proxy1.recv(), proxy2.recv(), return_exceptions=True)
            self.assertIsInstance(exc1, MediaStreamError)
            self.assertIsInstance(frame2, av.AudioFrame)
        source.stop()

    @asynctest
    async def test_audio_stop_consumer_unbuffered(self):
        source = AudioStreamTrack()
        relay = MediaRelay()
        proxy1 = relay.subscribe(source, buffered=False)
        proxy2 = relay.subscribe(source, buffered=False)
        samples_per_frame = 160
        for pts in range(0, 2 * samples_per_frame, samples_per_frame):
            (frame1, frame2) = await asyncio.gather(proxy1.recv(), proxy2.recv())
            self.assertEqual(frame1.format.name, 's16')
            self.assertEqual(frame1.layout.name, 'mono')
            self.assertEqual(frame1.pts, pts)
            self.assertEqual(frame1.samples, samples_per_frame)
            self.assertEqual(frame2.format.name, 's16')
            self.assertEqual(frame2.layout.name, 'mono')
            self.assertEqual(frame2.pts, pts)
            self.assertEqual(frame2.samples, samples_per_frame)
        proxy1.stop()
        for i in range(2):
            (exc1, frame2) = await asyncio.gather(proxy1.recv(), proxy2.recv(), return_exceptions=True)
            self.assertIsInstance(exc1, MediaStreamError)
            self.assertIsInstance(frame2, av.AudioFrame)
        source.stop()

    @asynctest
    async def test_audio_stop_source(self):
        source = AudioStreamTrack()
        relay = MediaRelay()
        proxy1 = relay.subscribe(source)
        proxy2 = relay.subscribe(source)
        samples_per_frame = 160
        for pts in range(0, 2 * samples_per_frame, samples_per_frame):
            (frame1, frame2) = await asyncio.gather(proxy1.recv(), proxy2.recv())
            self.assertEqual(frame1.format.name, 's16')
            self.assertEqual(frame1.layout.name, 'mono')
            self.assertEqual(frame1.pts, pts)
            self.assertEqual(frame1.samples, samples_per_frame)
            self.assertEqual(frame2.format.name, 's16')
            self.assertEqual(frame2.layout.name, 'mono')
            self.assertEqual(frame2.pts, pts)
            self.assertEqual(frame2.samples, samples_per_frame)
        source.stop()
        await asyncio.gather(proxy1.recv(), proxy2.recv())
        for i in range(2):
            (exc1, exc2) = await asyncio.gather(proxy1.recv(), proxy2.recv(), return_exceptions=True)
            self.assertIsInstance(exc1, MediaStreamError)
            self.assertIsInstance(exc2, MediaStreamError)

    @asynctest
    async def test_audio_stop_source_unbuffered(self):
        source = AudioStreamTrack()
        relay = MediaRelay()
        proxy1 = relay.subscribe(source, buffered=False)
        proxy2 = relay.subscribe(source, buffered=False)
        samples_per_frame = 160
        for pts in range(0, 2 * samples_per_frame, samples_per_frame):
            (frame1, frame2) = await asyncio.gather(proxy1.recv(), proxy2.recv())
            self.assertEqual(frame1.format.name, 's16')
            self.assertEqual(frame1.layout.name, 'mono')
            self.assertEqual(frame1.pts, pts)
            self.assertEqual(frame1.samples, samples_per_frame)
            self.assertEqual(frame2.format.name, 's16')
            self.assertEqual(frame2.layout.name, 'mono')
            self.assertEqual(frame2.pts, pts)
            self.assertEqual(frame2.samples, samples_per_frame)
        source.stop()
        for i in range(2):
            (exc1, exc2) = await asyncio.gather(proxy1.recv(), proxy2.recv(), return_exceptions=True)
            self.assertIsInstance(exc1, MediaStreamError)
            self.assertIsInstance(exc2, MediaStreamError)

    @asynctest
    async def test_audio_slow_consumer(self):
        source = AudioStreamTrack()
        relay = MediaRelay()
        proxy1 = relay.subscribe(source, buffered=False)
        proxy2 = relay.subscribe(source, buffered=False)
        samples_per_frame = 160
        for pts in range(0, 2 * samples_per_frame, samples_per_frame):
            (frame1, frame2) = await asyncio.gather(proxy1.recv(), proxy2.recv())
            self.assertEqual(frame1.format.name, 's16')
            self.assertEqual(frame1.layout.name, 'mono')
            self.assertEqual(frame1.pts, pts)
            self.assertEqual(frame1.samples, samples_per_frame)
            self.assertEqual(frame2.format.name, 's16')
            self.assertEqual(frame2.layout.name, 'mono')
            self.assertEqual(frame2.pts, pts)
            self.assertEqual(frame2.samples, samples_per_frame)
        timestamp = 5 * samples_per_frame
        await asyncio.sleep(source._start + timestamp / 8000 - time.time())
        (frame1, frame2) = await asyncio.gather(proxy1.recv(), proxy2.recv())
        self.assertEqual(frame1.format.name, 's16')
        self.assertEqual(frame1.layout.name, 'mono')
        self.assertEqual(frame1.pts, 5 * samples_per_frame)
        self.assertEqual(frame1.samples, samples_per_frame)
        self.assertEqual(frame2.format.name, 's16')
        self.assertEqual(frame2.layout.name, 'mono')
        self.assertEqual(frame2.pts, 5 * samples_per_frame)
        self.assertEqual(frame2.samples, samples_per_frame)
        proxy1.stop()
        for i in range(2):
            (exc1, frame2) = await asyncio.gather(proxy1.recv(), proxy2.recv(), return_exceptions=True)
            self.assertIsInstance(exc1, MediaStreamError)
            self.assertIsInstance(frame2, av.AudioFrame)
        source.stop()

class BufferingInputContainer:

    def __init__(self, real):
        if False:
            while True:
                i = 10
        self.__failed = False
        self.__real = real

    def decode(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not self.__failed:
            self.__failed = True
            raise av.AVError(errno.EAGAIN, 'EAGAIN')
        return self.__real.decode(*args, **kwargs)

    def demux(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not self.__failed:
            self.__failed = True
            raise av.AVError(errno.EAGAIN, 'EAGAIN')
        return self.__real.demux(*args, **kwargs)

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        return getattr(self.__real, name)

class MediaPlayerTest(MediaTestCase):

    def assertAudio(self, frame):
        if False:
            while True:
                i = 10
        self.assertEqual(frame.format.name, 's16')
        self.assertEqual(frame.layout.name, 'stereo')
        self.assertEqual(frame.samples, 960)
        self.assertEqual(frame.sample_rate, 48000)

    def assertVideo(self, frame):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(frame.width, 640)
        self.assertEqual(frame.height, 480)

    def createMediaPlayer(self, path, **kwargs):
        if False:
            return 10
        return MediaPlayer(path, **kwargs)

    def endTime(self, frame):
        if False:
            return 10
        return frame.time + frame.samples / frame.sample_rate

    @asynctest
    async def test_audio_file_8kHz(self):
        path = self.create_audio_file('test.wav')
        player = self.createMediaPlayer(path)
        if isinstance(self, MediaPlayerNoDecodeTest):
            self.assertIsNone(player.audio)
            self.assertIsNone(player.video)
        else:
            self.assertIsNotNone(player.audio)
            self.assertIsNone(player.video)
            self.assertEqual(player.audio.readyState, 'live')
            while True:
                frame = await player.audio.recv()
                self.assertAudio(frame)
                if self.endTime(frame) >= 0.98:
                    break
            with self.assertRaises(MediaStreamError):
                await player.audio.recv()
            self.assertEqual(player.audio.readyState, 'ended')
            with self.assertRaises(MediaStreamError):
                await player.audio.recv()

    @asynctest
    async def test_audio_file_48kHz(self):
        path = self.create_audio_file('test.wav', sample_rate=48000)
        player = self.createMediaPlayer(path)
        if isinstance(self, MediaPlayerNoDecodeTest):
            self.assertIsNone(player.audio)
            self.assertIsNone(player.video)
        else:
            self.assertIsNotNone(player.audio)
            self.assertIsNone(player.video)
            self.assertEqual(player.audio.readyState, 'live')
            while True:
                frame = await player.audio.recv()
                if self.endTime(frame) >= 1.0:
                    break
                self.assertAudio(frame)
            with self.assertRaises(MediaStreamError):
                await player.audio.recv()
            self.assertEqual(player.audio.readyState, 'ended')

    @asynctest
    async def test_audio_file_looping(self):
        path = self.create_audio_file('test.wav', sample_rate=48000)
        player = self.createMediaPlayer(path, loop=True)
        if isinstance(self, MediaPlayerNoDecodeTest):
            self.assertIsNone(player.audio)
        else:
            self.assertEqual(player.audio.readyState, 'live')
            for i in range(100):
                frame = await player.audio.recv()
                self.assertAudio(frame)
            await player.audio.recv()
            self.assertEqual(player.audio.readyState, 'live')
            player.audio.stop()

    @asynctest
    async def test_audio_and_video_file(self):
        path = self.create_audio_and_video_file(name='test.ts', duration=5)
        player = self.createMediaPlayer(path)
        self.assertIsNotNone(player.audio)
        self.assertIsNotNone(player.video)
        self.assertEqual(player.audio.readyState, 'live')
        self.assertEqual(player.video.readyState, 'live')
        for i in range(10):
            await asyncio.gather(player.audio.recv(), player.video.recv())
        player.audio.stop()
        for i in range(10):
            with self.assertRaises(MediaStreamError):
                await player.audio.recv()
            await player.video.recv()
        player.video.stop()
        with self.assertRaises(MediaStreamError):
            await player.audio.recv()
        with self.assertRaises(MediaStreamError):
            await player.video.recv()

    @asynctest
    async def test_video_file_mp4(self):
        path = self.create_video_file('test.mp4', duration=3)
        player = self.createMediaPlayer(path)
        if isinstance(self, MediaPlayerNoDecodeTest):
            self.assertIsNone(player.audio)
            self.assertIsNone(player.video)
        else:
            self.assertIsNone(player.audio)
            self.assertIsNotNone(player.video)
            self.assertEqual(player.video.readyState, 'live')
            for i in range(90):
                frame = await player.video.recv()
                self.assertVideo(frame)
            with self.assertRaises(MediaStreamError):
                await player.video.recv()
            self.assertEqual(player.video.readyState, 'ended')

    @asynctest
    async def test_audio_and_video_file_mpegts_eagain(self):
        path = self.create_audio_and_video_file('test.ts', duration=3)
        container = BufferingInputContainer(av.open(path, 'r'))
        with patch('av.open') as mock_open:
            mock_open.return_value = container
            player = self.createMediaPlayer(path)
        self.assertIsNotNone(player.audio)
        self.assertIsNotNone(player.video)
        self.assertEqual(player.video.readyState, 'live')
        error_count = 0
        received_count = 0
        for i in range(100):
            try:
                frame = await player.video.recv()
                self.assertVideo(frame)
                received_count += 1
            except MediaStreamError:
                error_count += 1
                break
        self.assertEqual(error_count, 1)
        self.assertGreaterEqual(received_count, 89)
        self.assertEqual(player.video.readyState, 'ended')

    @asynctest
    async def test_video_file_mpegts_looping(self):
        path = self.create_video_file('test.ts', duration=5)
        player = self.createMediaPlayer(path, loop=True)
        self.assertEqual(player.video.readyState, 'live')
        for i in range(100):
            frame = await player.video.recv()
            self.assertVideo(frame)
        await player.video.recv()
        self.assertEqual(player.video.readyState, 'live')
        player.video.stop()

    @asynctest
    async def test_video_file_png(self):
        path = self.create_video_file('test-%3d.png', duration=3)
        player = self.createMediaPlayer(path)
        if isinstance(self, MediaPlayerNoDecodeTest):
            self.assertIsNone(player.audio)
            self.assertIsNone(player.video)
        else:
            self.assertIsNone(player.audio)
            self.assertIsNotNone(player.video)
            self.assertEqual(player.video.readyState, 'live')
            for i in range(90):
                frame = await player.video.recv()
                self.assertVideo(frame)
            with self.assertRaises(MediaStreamError):
                await player.video.recv()
            self.assertEqual(player.video.readyState, 'ended')

class MediaPlayerNoDecodeTest(MediaPlayerTest):

    def assertAudio(self, packet):
        if False:
            while True:
                i = 10
        self.assertIsInstance(packet, av.Packet)

    def assertVideo(self, packet):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(packet, av.Packet)

    def createMediaPlayer(self, path, **kwargs):
        if False:
            return 10
        return MediaPlayer(path, decode=False, **kwargs)

    def endTime(self, packet):
        if False:
            return 10
        return float((packet.pts + packet.duration) * packet.time_base)

class MediaRecorderTest(MediaTestCase):

    @asynctest
    async def test_audio_mp3(self):
        path = self.temporary_path('test.mp3')
        recorder = MediaRecorder(path)
        recorder.addTrack(AudioStreamTrack())
        await recorder.start()
        await asyncio.sleep(2)
        await recorder.stop()
        container = av.open(path, 'r')
        self.assertEqual(len(container.streams), 1)
        self.assertIn(container.streams[0].codec.name, ('mp3', 'mp3float'))
        self.assertGreater(float(container.streams[0].duration * container.streams[0].time_base), 0)

    @asynctest
    async def test_audio_wav(self):
        path = self.temporary_path('test.wav')
        recorder = MediaRecorder(path)
        recorder.addTrack(AudioStreamTrack())
        await recorder.start()
        await asyncio.sleep(2)
        await recorder.stop()
        container = av.open(path, 'r')
        self.assertEqual(len(container.streams), 1)
        self.assertEqual(container.streams[0].codec.name, 'pcm_s16le')
        self.assertGreater(float(container.streams[0].duration * container.streams[0].time_base), 0)

    @asynctest
    async def test_audio_wav_ended(self):
        track = AudioStreamTrack()
        recorder = MediaRecorder(self.temporary_path('test.wav'))
        recorder.addTrack(track)
        await recorder.start()
        await asyncio.sleep(1)
        track.stop()
        await asyncio.sleep(1)
        await recorder.stop()

    @asynctest
    async def test_audio_and_video(self):
        path = self.temporary_path('test.mp4')
        recorder = MediaRecorder(path)
        recorder.addTrack(AudioStreamTrack())
        recorder.addTrack(VideoStreamTrack())
        await recorder.start()
        await asyncio.sleep(2)
        await recorder.stop()
        container = av.open(path, 'r')
        self.assertEqual(len(container.streams), 2)
        self.assertEqual(container.streams[0].codec.name, 'aac')
        self.assertGreater(float(container.streams[0].duration * container.streams[0].time_base), 0)
        self.assertEqual(container.streams[1].codec.name, 'h264')
        self.assertEqual(container.streams[1].width, 640)
        self.assertEqual(container.streams[1].height, 480)
        self.assertGreater(float(container.streams[1].duration * container.streams[1].time_base), 0)

    @asynctest
    async def test_video_png(self):
        path = self.temporary_path('test-%3d.png')
        recorder = MediaRecorder(path)
        recorder.addTrack(VideoStreamTrack())
        await recorder.start()
        await asyncio.sleep(2)
        await recorder.stop()
        container = av.open(path, 'r')
        self.assertEqual(len(container.streams), 1)
        self.assertEqual(container.streams[0].codec.name, 'png')
        self.assertGreater(float(container.streams[0].duration * container.streams[0].time_base), 0)
        self.assertEqual(container.streams[0].width, 640)
        self.assertEqual(container.streams[0].height, 480)

    @asynctest
    async def test_video_mp4(self):
        path = self.temporary_path('test.mp4')
        recorder = MediaRecorder(path)
        recorder.addTrack(VideoStreamTrack())
        await recorder.start()
        await asyncio.sleep(2)
        await recorder.stop()
        container = av.open(path, 'r')
        self.assertEqual(len(container.streams), 1)
        self.assertEqual(container.streams[0].codec.name, 'h264')
        self.assertGreater(float(container.streams[0].duration * container.streams[0].time_base), 0)
        self.assertEqual(container.streams[0].width, 640)
        self.assertEqual(container.streams[0].height, 480)

    @asynctest
    async def test_video_mp4_uhd(self):
        path = self.temporary_path('test.mp4')
        recorder = MediaRecorder(path)
        recorder.addTrack(VideoStreamTrackUhd())
        await recorder.start()
        await asyncio.sleep(2)
        await recorder.stop()
        container = av.open(path, 'r')
        self.assertEqual(len(container.streams), 1)
        self.assertEqual(container.streams[0].codec.name, 'h264')
        self.assertGreater(float(container.streams[0].duration * container.streams[0].time_base), 0)
        self.assertEqual(container.streams[0].width, 3840)
        self.assertEqual(container.streams[0].height, 2160)