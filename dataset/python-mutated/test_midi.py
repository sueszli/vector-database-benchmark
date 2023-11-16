from picard.formats import midi
from .common import TAGS, CommonTests

class MIDITest(CommonTests.SimpleFormatsTestCase):
    testfile = 'test.mid'
    expected_info = {'length': 127997, '~format': 'Standard MIDI File'}
    unexpected_info = ['~video']

    def test_supports_tag(self):
        if False:
            i = 10
            return i + 15
        for tag in TAGS:
            self.assertFalse(midi.MIDIFile.supports_tag(tag))