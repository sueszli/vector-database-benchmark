from unittest import TestCase
from aiortc.jitterbuffer import JitterBuffer
from aiortc.rtp import RtpPacket

class JitterBufferTest(TestCase):

    def assertPackets(self, jbuffer, expected):
        if False:
            i = 10
            return i + 15
        found = [x.sequence_number if x else None for x in jbuffer._packets]
        self.assertEqual(found, expected)

    def test_create(self):
        if False:
            i = 10
            return i + 15
        jbuffer = JitterBuffer(capacity=2)
        self.assertEqual(jbuffer._packets, [None, None])
        self.assertEqual(jbuffer._origin, None)
        jbuffer = JitterBuffer(capacity=4)
        self.assertEqual(jbuffer._packets, [None, None, None, None])
        self.assertEqual(jbuffer._origin, None)

    def test_add_ordered(self):
        if False:
            return 10
        jbuffer = JitterBuffer(capacity=4)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [0, None, None, None])
        self.assertEqual(jbuffer._origin, 0)
        self.assertFalse(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [0, 1, None, None])
        self.assertEqual(jbuffer._origin, 0)
        self.assertFalse(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [0, 1, 2, None])
        self.assertEqual(jbuffer._origin, 0)
        self.assertFalse(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [0, 1, 2, 3])
        self.assertEqual(jbuffer._origin, 0)
        self.assertFalse(pli_flag)

    def test_add_unordered(self):
        if False:
            print('Hello World!')
        jbuffer = JitterBuffer(capacity=4)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, 1, None, None])
        self.assertEqual(jbuffer._origin, 1)
        self.assertFalse(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, 1, None, 3])
        self.assertEqual(jbuffer._origin, 1)
        self.assertFalse(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, 1, 2, 3])
        self.assertEqual(jbuffer._origin, 1)
        self.assertFalse(pli_flag)

    def test_add_seq_too_low_drop(self):
        if False:
            i = 10
            return i + 15
        jbuffer = JitterBuffer(capacity=4)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, None, 2, None])
        self.assertEqual(jbuffer._origin, 2)
        self.assertFalse(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, None, 2, None])
        self.assertEqual(jbuffer._origin, 2)
        self.assertFalse(pli_flag)

    def test_add_seq_too_low_reset(self):
        if False:
            for i in range(10):
                print('nop')
        jbuffer = JitterBuffer(capacity=4)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=2000, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [2000, None, None, None])
        self.assertEqual(jbuffer._origin, 2000)
        self.assertFalse(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertIsNone(frame)
        self.assertPackets(jbuffer, [None, 1, None, None])
        self.assertEqual(jbuffer._origin, 1)
        self.assertFalse(pli_flag)

    def test_add_seq_too_high_discard_one(self):
        if False:
            while True:
                i = 10
        jbuffer = JitterBuffer(capacity=4)
        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=4, timestamp=1234))
        self.assertEqual(jbuffer._origin, 4)
        self.assertPackets(jbuffer, [4, None, None, None])

    def test_add_seq_too_high_discard_one_v2(self):
        if False:
            print('Hello World!')
        jbuffer = JitterBuffer(capacity=4)
        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1235))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=4, timestamp=1235))
        self.assertEqual(jbuffer._origin, 3)
        self.assertPackets(jbuffer, [4, None, None, 3])

    def test_add_seq_too_high_discard_four(self):
        if False:
            for i in range(10):
                print('nop')
        jbuffer = JitterBuffer(capacity=4)
        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=7, timestamp=1235))
        self.assertEqual(jbuffer._origin, 7)
        self.assertPackets(jbuffer, [None, None, None, 7])

    def test_add_seq_too_high_discard_more(self):
        if False:
            for i in range(10):
                print('nop')
        jbuffer = JitterBuffer(capacity=4)
        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        jbuffer.add(RtpPacket(sequence_number=8, timestamp=1234))
        self.assertEqual(jbuffer._origin, 8)
        self.assertPackets(jbuffer, [8, None, None, None])

    def test_add_seq_too_high_reset(self):
        if False:
            return 10
        jbuffer = JitterBuffer(capacity=4)
        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        self.assertPackets(jbuffer, [0, None, None, None])
        jbuffer.add(RtpPacket(sequence_number=3000, timestamp=1234))
        self.assertEqual(jbuffer._origin, 3000)
        self.assertPackets(jbuffer, [3000, None, None, None])

    def test_remove(self):
        if False:
            while True:
                i = 10
        jbuffer = JitterBuffer(capacity=4)
        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        jbuffer.add(RtpPacket(sequence_number=2, timestamp=1234))
        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1234))
        self.assertEqual(jbuffer._origin, 0)
        self.assertPackets(jbuffer, [0, 1, 2, 3])
        jbuffer.remove(1)
        self.assertEqual(jbuffer._origin, 1)
        self.assertPackets(jbuffer, [None, 1, 2, 3])
        jbuffer.remove(2)
        self.assertEqual(jbuffer._origin, 3)
        self.assertPackets(jbuffer, [None, None, None, 3])

    def test_smart_remove(self):
        if False:
            while True:
                i = 10
        jbuffer = JitterBuffer(capacity=4)
        jbuffer.add(RtpPacket(sequence_number=0, timestamp=1234))
        jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        jbuffer.add(RtpPacket(sequence_number=3, timestamp=1235))
        self.assertEqual(jbuffer._origin, 0)
        self.assertPackets(jbuffer, [0, 1, None, 3])
        jbuffer.smart_remove(1)
        self.assertEqual(jbuffer._origin, 3)
        self.assertPackets(jbuffer, [None, None, None, 3])

    def test_remove_audio_frame(self):
        if False:
            while True:
                i = 10
        '\n        Audio jitter buffer.\n        '
        jbuffer = JitterBuffer(capacity=16, prefetch=4)
        packet = RtpPacket(sequence_number=0, timestamp=1234)
        packet._data = b'0000'
        (pli_flag, frame) = jbuffer.add(packet)
        self.assertIsNone(frame)
        packet = RtpPacket(sequence_number=1, timestamp=1235)
        packet._data = b'0001'
        (pli_flag, frame) = jbuffer.add(packet)
        self.assertIsNone(frame)
        packet = RtpPacket(sequence_number=2, timestamp=1236)
        packet._data = b'0002'
        (pli_flag, frame) = jbuffer.add(packet)
        self.assertIsNone(frame)
        packet = RtpPacket(sequence_number=3, timestamp=1237)
        packet._data = b'0003'
        (pli_flag, frame) = jbuffer.add(packet)
        self.assertIsNone(frame)
        packet = RtpPacket(sequence_number=4, timestamp=1238)
        packet._data = b'0003'
        (pli_flag, frame) = jbuffer.add(packet)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.data, b'0000')
        self.assertEqual(frame.timestamp, 1234)
        packet = RtpPacket(sequence_number=5, timestamp=1239)
        packet._data = b'0004'
        (pli_flag, frame) = jbuffer.add(packet)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.data, b'0001')
        self.assertEqual(frame.timestamp, 1235)

    def test_remove_video_frame(self):
        if False:
            i = 10
            return i + 15
        '\n        Video jitter buffer.\n        '
        jbuffer = JitterBuffer(capacity=128, is_video=True)
        packet = RtpPacket(sequence_number=0, timestamp=1234)
        packet._data = b'0000'
        (pli_flag, frame) = jbuffer.add(packet)
        self.assertIsNone(frame)
        packet = RtpPacket(sequence_number=1, timestamp=1234)
        packet._data = b'0001'
        (pli_flag, frame) = jbuffer.add(packet)
        self.assertIsNone(frame)
        packet = RtpPacket(sequence_number=2, timestamp=1234)
        packet._data = b'0002'
        (pli_flag, frame) = jbuffer.add(packet)
        self.assertIsNone(frame)
        packet = RtpPacket(sequence_number=3, timestamp=1235)
        packet._data = b'0003'
        (pli_flag, frame) = jbuffer.add(packet)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.data, b'000000010002')
        self.assertEqual(frame.timestamp, 1234)

    def test_pli_flag(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Video jitter buffer.\n        '
        jbuffer = JitterBuffer(capacity=128, is_video=True)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=2000, timestamp=1234))
        self.assertIsNone(frame)
        self.assertEqual(jbuffer._origin, 2000)
        self.assertFalse(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=1, timestamp=1234))
        self.assertIsNone(frame)
        self.assertEqual(jbuffer._origin, 1)
        self.assertTrue(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=128, timestamp=1235))
        self.assertIsNone(frame)
        self.assertEqual(jbuffer._origin, 1)
        self.assertFalse(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=129, timestamp=1235))
        self.assertIsNone(frame)
        self.assertEqual(jbuffer._origin, 128)
        self.assertTrue(pli_flag)
        (pli_flag, frame) = jbuffer.add(RtpPacket(sequence_number=2000, timestamp=2345))
        self.assertIsNone(frame)
        self.assertEqual(jbuffer._origin, 2000)
        self.assertTrue(pli_flag)