import unittest
from test.helper import TestHelper
from unittest.mock import ANY, Mock, call, patch
from beets import util
from beets.library import Item
from beetsplug.mpdstats import MPDStats

class MPDStatsTest(unittest.TestCase, TestHelper):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_beets()
        self.load_plugins('mpdstats')

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.teardown_beets()
        self.unload_plugins()

    def test_update_rating(self):
        if False:
            print('Hello World!')
        item = Item(title='title', path='', id=1)
        item.add(self.lib)
        log = Mock()
        mpdstats = MPDStats(self.lib, log)
        self.assertFalse(mpdstats.update_rating(item, True))
        self.assertFalse(mpdstats.update_rating(None, True))

    def test_get_item(self):
        if False:
            while True:
                i = 10
        item_path = util.normpath('/foo/bar.flac')
        item = Item(title='title', path=item_path, id=1)
        item.add(self.lib)
        log = Mock()
        mpdstats = MPDStats(self.lib, log)
        self.assertEqual(str(mpdstats.get_item(item_path)), str(item))
        self.assertIsNone(mpdstats.get_item('/some/non-existing/path'))
        self.assertIn('item not found:', log.info.call_args[0][0])
    FAKE_UNKNOWN_STATE = 'some-unknown-one'
    STATUSES = [{'state': FAKE_UNKNOWN_STATE}, {'state': 'pause'}, {'state': 'play', 'songid': 1, 'time': '0:1'}, {'state': 'stop'}]
    EVENTS = [['player']] * (len(STATUSES) - 1) + [KeyboardInterrupt]
    item_path = util.normpath('/foo/bar.flac')
    songid = 1

    @patch('beetsplug.mpdstats.MPDClientWrapper', return_value=Mock(**{'events.side_effect': EVENTS, 'status.side_effect': STATUSES, 'currentsong.return_value': (item_path, songid)}))
    def test_run_mpdstats(self, mpd_mock):
        if False:
            print('Hello World!')
        item = Item(title='title', path=self.item_path, id=1)
        item.add(self.lib)
        log = Mock()
        try:
            MPDStats(self.lib, log).run()
        except KeyboardInterrupt:
            pass
        log.debug.assert_has_calls([call('unhandled status "{0}"', ANY)])
        log.info.assert_has_calls([call('pause'), call('playing {0}', ANY), call('stop')])

def suite():
    if False:
        while True:
            i = 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')