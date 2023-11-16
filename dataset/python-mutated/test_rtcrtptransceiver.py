from unittest import TestCase
from aiortc.rtcrtpparameters import RTCRtpCodecCapability
from aiortc.rtcrtptransceiver import RTCRtpTransceiver

class RTCRtpTransceiverTest(TestCase):

    def test_codec_preferences(self):
        if False:
            i = 10
            return i + 15
        transceiver = RTCRtpTransceiver('audio', None, None)
        self.assertEqual(transceiver._preferred_codecs, [])
        transceiver.setCodecPreferences([])
        self.assertEqual(transceiver._preferred_codecs, [])
        transceiver.setCodecPreferences([RTCRtpCodecCapability(mimeType='audio/PCMU', clockRate=8000, channels=1)])
        self.assertEqual(transceiver._preferred_codecs, [RTCRtpCodecCapability(mimeType='audio/PCMU', clockRate=8000, channels=1)])
        transceiver.setCodecPreferences([RTCRtpCodecCapability(mimeType='audio/PCMU', clockRate=8000, channels=1), RTCRtpCodecCapability(mimeType='audio/PCMU', clockRate=8000, channels=1)])
        self.assertEqual(transceiver._preferred_codecs, [RTCRtpCodecCapability(mimeType='audio/PCMU', clockRate=8000, channels=1)])
        with self.assertRaises(ValueError) as cm:
            transceiver.setCodecPreferences([RTCRtpCodecCapability(mimeType='audio/bogus', clockRate=8000, channels=1)])
        self.assertEqual(str(cm.exception), 'Codec is not in capabilities')