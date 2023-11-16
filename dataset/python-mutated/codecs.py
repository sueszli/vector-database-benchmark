import fractions
from unittest import TestCase
from aiortc.codecs import depayload, get_decoder, get_encoder
from aiortc.jitterbuffer import JitterFrame
from aiortc.mediastreams import AUDIO_PTIME, VIDEO_TIME_BASE
from av import AudioFrame, VideoFrame
from av.packet import Packet

class CodecTestCase(TestCase):

    def create_audio_frame(self, samples, pts, layout='mono', sample_rate=48000):
        if False:
            print('Hello World!')
        frame = AudioFrame(format='s16', layout=layout, samples=samples)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = pts
        frame.sample_rate = sample_rate
        frame.time_base = fractions.Fraction(1, sample_rate)
        return frame

    def create_audio_frames(self, layout, sample_rate, count):
        if False:
            i = 10
            return i + 15
        frames = []
        timestamp = 0
        samples_per_frame = int(AUDIO_PTIME * sample_rate)
        for i in range(count):
            frames.append(self.create_audio_frame(samples=samples_per_frame, pts=timestamp, layout=layout, sample_rate=sample_rate))
            timestamp += samples_per_frame
        return frames

    def create_packet(self, payload: bytes, pts: int) -> Packet:
        if False:
            return 10
        '\n        Create a packet.\n        '
        packet = Packet(len(payload))
        packet.update(payload)
        packet.pts = pts
        packet.time_base = fractions.Fraction(1, 1000)
        return packet

    def create_video_frame(self, width, height, pts, format='yuv420p', time_base=VIDEO_TIME_BASE):
        if False:
            while True:
                i = 10
        '\n        Create a single blank video frame.\n        '
        frame = VideoFrame(width=width, height=height, format=format)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def create_video_frames(self, width, height, count, time_base=VIDEO_TIME_BASE):
        if False:
            i = 10
            return i + 15
        '\n        Create consecutive blank video frames.\n        '
        frames = []
        for i in range(count):
            frames.append(self.create_video_frame(width=width, height=height, pts=int(i / time_base / 30), time_base=time_base))
        return frames

    def roundtrip_audio(self, codec, output_layout, output_sample_rate, drop=[]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Round-trip an AudioFrame through encoder then decoder.\n        '
        encoder = get_encoder(codec)
        decoder = get_decoder(codec)
        input_frames = self.create_audio_frames(layout='mono', sample_rate=8000, count=10)
        output_sample_count = int(output_sample_rate * AUDIO_PTIME)
        for (i, frame) in enumerate(input_frames):
            (packages, timestamp) = encoder.encode(frame)
            if i not in drop:
                data = b''
                for package in packages:
                    data += depayload(codec, package)
                frames = decoder.decode(JitterFrame(data=data, timestamp=timestamp))
                self.assertEqual(len(frames), 1)
                self.assertEqual(frames[0].format.name, 's16')
                self.assertEqual(frames[0].layout.name, output_layout)
                self.assertEqual(frames[0].samples, output_sample_rate * AUDIO_PTIME)
                self.assertEqual(frames[0].sample_rate, output_sample_rate)
                self.assertEqual(frames[0].pts, i * output_sample_count)
                self.assertEqual(frames[0].time_base, fractions.Fraction(1, output_sample_rate))

    def roundtrip_video(self, codec, width, height, time_base=VIDEO_TIME_BASE):
        if False:
            for i in range(10):
                print('nop')
        '\n        Round-trip a VideoFrame through encoder then decoder.\n        '
        encoder = get_encoder(codec)
        decoder = get_decoder(codec)
        input_frames = self.create_video_frames(width=width, height=height, count=30, time_base=time_base)
        for (i, frame) in enumerate(input_frames):
            (packages, timestamp) = encoder.encode(frame)
            data = b''
            for package in packages:
                data += depayload(codec, package)
            frames = decoder.decode(JitterFrame(data=data, timestamp=timestamp))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].width, frame.width)
            self.assertEqual(frames[0].height, frame.height)
            self.assertEqual(frames[0].pts, i * 3000)
            self.assertEqual(frames[0].time_base, VIDEO_TIME_BASE)