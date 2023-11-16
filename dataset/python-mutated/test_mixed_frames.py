"""Test pulse logical elements and frames"""
from qiskit.pulse import Port, Qubit, GenericFrame, MixedFrame
from qiskit.test import QiskitTestCase

class TestMixedFrames(QiskitTestCase):
    """Test mixed frames."""

    def test_mixed_frame_initialization(self):
        if False:
            while True:
                i = 10
        'Test that MixedFrame objects are created correctly'
        frame = GenericFrame('frame1')
        qubit = Qubit(1)
        mixed_frame = MixedFrame(qubit, frame)
        self.assertEqual(mixed_frame.pulse_target, qubit)
        self.assertEqual(mixed_frame.frame, frame)
        port = Port('d0')
        mixed_frame = MixedFrame(port, frame)
        self.assertEqual(mixed_frame.pulse_target, port)

    def test_mixed_frames_comparison(self):
        if False:
            i = 10
            return i + 15
        'Test the comparison of various mixed frames'
        self.assertEqual(MixedFrame(Qubit(1), GenericFrame('a')), MixedFrame(Qubit(1), GenericFrame('a')))
        self.assertEqual(MixedFrame(Port('s'), GenericFrame('a')), MixedFrame(Port('s'), GenericFrame('a')))
        self.assertNotEqual(MixedFrame(Qubit(1), GenericFrame('a')), MixedFrame(Qubit(2), GenericFrame('a')))
        self.assertNotEqual(MixedFrame(Qubit(1), GenericFrame('a')), MixedFrame(Qubit(1), GenericFrame('b')))

    def test_mixed_frame_repr(self):
        if False:
            for i in range(10):
                print('nop')
        'Test MixedFrame __repr__'
        frame = GenericFrame('frame1')
        qubit = Qubit(1)
        mixed_frame = MixedFrame(qubit, frame)
        self.assertEqual(str(mixed_frame), f'MixedFrame({qubit},{frame})')