from unittest import TestCase
from aiortc import RTCSessionDescription

class RTCSessionDescriptionTest(TestCase):

    def test_bad_type(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError) as cm:
            RTCSessionDescription(sdp='v=0\r\n', type='bogus')
        self.assertEqual(str(cm.exception), "'type' must be in ['offer', 'pranswer', 'answer', 'rollback'] (got 'bogus')")

    def test_good_type(self):
        if False:
            return 10
        desc = RTCSessionDescription(sdp='v=0\r\n', type='answer')
        self.assertEqual(desc.sdp, 'v=0\r\n')
        self.assertEqual(desc.type, 'answer')