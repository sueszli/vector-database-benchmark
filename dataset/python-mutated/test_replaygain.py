import unittest
from test.helper import TestHelper, has_program
from mediafile import MediaFile
from beets import config
from beetsplug.replaygain import FatalGstreamerPluginReplayGainError, GStreamerBackend
try:
    import gi
    gi.require_version('Gst', '1.0')
    GST_AVAILABLE = True
except (ImportError, ValueError):
    GST_AVAILABLE = False
if any((has_program(cmd, ['-v']) for cmd in ['mp3gain', 'aacgain'])):
    GAIN_PROG_AVAILABLE = True
else:
    GAIN_PROG_AVAILABLE = False
FFMPEG_AVAILABLE = has_program('ffmpeg', ['-version'])

def reset_replaygain(item):
    if False:
        print('Hello World!')
    item['rg_track_peak'] = None
    item['rg_track_gain'] = None
    item['rg_album_gain'] = None
    item['rg_album_gain'] = None
    item['r128_track_gain'] = None
    item['r128_album_gain'] = None
    item.write()
    item.store()

class GstBackendMixin:
    backend = 'gstreamer'
    has_r128_support = True

    def test_backend(self):
        if False:
            for i in range(10):
                print('nop')
        'Check whether the backend actually has all required functionality.'
        try:
            config['replaygain']['targetlevel'] = 89
            GStreamerBackend(config['replaygain'], None)
        except FatalGstreamerPluginReplayGainError as e:
            self.skipTest(str(e))

class CmdBackendMixin:
    backend = 'command'
    has_r128_support = False

    def test_backend(self):
        if False:
            i = 10
            return i + 15
        'Check whether the backend actually has all required functionality.'
        pass

class FfmpegBackendMixin:
    backend = 'ffmpeg'
    has_r128_support = True

    def test_backend(self):
        if False:
            for i in range(10):
                print('nop')
        'Check whether the backend actually has all required functionality.'
        pass

class ReplayGainCliTestBase(TestHelper):
    FNAME: str

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.test_backend()
        self.setup_beets(disk=True)
        self.config['replaygain']['backend'] = self.backend
        try:
            self.load_plugins('replaygain')
        except Exception:
            self.teardown_beets()
            self.unload_plugins()

    def _add_album(self, *args, **kwargs):
        if False:
            print('Hello World!')
        album = self.add_album_fixture(*args, fname=self.FNAME, **kwargs)
        for item in album.items():
            reset_replaygain(item)
        return album

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.teardown_beets()
        self.unload_plugins()

    def test_cli_saves_track_gain(self):
        if False:
            while True:
                i = 10
        self._add_album(2)
        for item in self.lib.items():
            self.assertIsNone(item.rg_track_peak)
            self.assertIsNone(item.rg_track_gain)
            mediafile = MediaFile(item.path)
            self.assertIsNone(mediafile.rg_track_peak)
            self.assertIsNone(mediafile.rg_track_gain)
        self.run_command('replaygain')
        if all((i.rg_track_peak is None and i.rg_track_gain is None for i in self.lib.items())):
            self.skipTest('decoder plugins could not be loaded.')
        for item in self.lib.items():
            self.assertIsNotNone(item.rg_track_peak)
            self.assertIsNotNone(item.rg_track_gain)
            mediafile = MediaFile(item.path)
            self.assertAlmostEqual(mediafile.rg_track_peak, item.rg_track_peak, places=6)
            self.assertAlmostEqual(mediafile.rg_track_gain, item.rg_track_gain, places=2)

    def test_cli_skips_calculated_tracks(self):
        if False:
            while True:
                i = 10
        album_rg = self._add_album(1)
        item_rg = album_rg.items()[0]
        if self.has_r128_support:
            album_r128 = self._add_album(1, ext='opus')
            item_r128 = album_r128.items()[0]
        self.run_command('replaygain')
        item_rg.load()
        self.assertIsNotNone(item_rg.rg_track_gain)
        self.assertIsNotNone(item_rg.rg_track_peak)
        self.assertIsNone(item_rg.r128_track_gain)
        item_rg.rg_track_gain += 1.0
        item_rg.rg_track_peak += 1.0
        item_rg.store()
        rg_track_gain = item_rg.rg_track_gain
        rg_track_peak = item_rg.rg_track_peak
        if self.has_r128_support:
            item_r128.load()
            self.assertIsNotNone(item_r128.r128_track_gain)
            self.assertIsNone(item_r128.rg_track_gain)
            self.assertIsNone(item_r128.rg_track_peak)
            item_r128.r128_track_gain += 1.0
            item_r128.store()
            r128_track_gain = item_r128.r128_track_gain
        self.run_command('replaygain')
        item_rg.load()
        self.assertEqual(item_rg.rg_track_gain, rg_track_gain)
        self.assertEqual(item_rg.rg_track_peak, rg_track_peak)
        if self.has_r128_support:
            item_r128.load()
            self.assertEqual(item_r128.r128_track_gain, r128_track_gain)

    def test_cli_does_not_skip_wrong_tag_type(self):
        if False:
            while True:
                i = 10
        "Check that items that have tags of the wrong type won't be skipped."
        if not self.has_r128_support:
            self.skipTest('r128 tags for opus not supported on backend {}'.format(self.backend))
        album_rg = self._add_album(1)
        item_rg = album_rg.items()[0]
        album_r128 = self._add_album(1, ext='opus')
        item_r128 = album_r128.items()[0]
        item_rg.r128_track_gain = 0.0
        item_rg.store()
        item_r128.rg_track_gain = 0.0
        item_r128.rg_track_peak = 42.0
        item_r128.store()
        self.run_command('replaygain')
        item_rg.load()
        item_r128.load()
        self.assertIsNotNone(item_rg.rg_track_gain)
        self.assertIsNotNone(item_rg.rg_track_peak)
        self.assertIsNotNone(item_r128.r128_track_gain)

    def test_cli_saves_album_gain_to_file(self):
        if False:
            while True:
                i = 10
        self._add_album(2)
        for item in self.lib.items():
            mediafile = MediaFile(item.path)
            self.assertIsNone(mediafile.rg_album_peak)
            self.assertIsNone(mediafile.rg_album_gain)
        self.run_command('replaygain', '-a')
        peaks = []
        gains = []
        for item in self.lib.items():
            mediafile = MediaFile(item.path)
            peaks.append(mediafile.rg_album_peak)
            gains.append(mediafile.rg_album_gain)
        self.assertEqual(max(peaks), min(peaks))
        self.assertEqual(max(gains), min(gains))
        self.assertNotEqual(max(gains), 0.0)
        self.assertNotEqual(max(peaks), 0.0)

    def test_cli_writes_only_r128_tags(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.has_r128_support:
            self.skipTest('r128 tags for opus not supported on backend {}'.format(self.backend))
        album = self._add_album(2, ext='opus')
        self.run_command('replaygain', '-a')
        for item in album.items():
            mediafile = MediaFile(item.path)
            self.assertIsNone(mediafile.rg_track_gain)
            self.assertIsNone(mediafile.rg_album_gain)
            self.assertIsNotNone(mediafile.r128_track_gain)
            self.assertIsNotNone(mediafile.r128_album_gain)

    def test_targetlevel_has_effect(self):
        if False:
            print('Hello World!')
        album = self._add_album(1)
        item = album.items()[0]

        def analyse(target_level):
            if False:
                i = 10
                return i + 15
            self.config['replaygain']['targetlevel'] = target_level
            self.run_command('replaygain', '-f')
            item.load()
            return item.rg_track_gain
        gain_relative_to_84 = analyse(84)
        gain_relative_to_89 = analyse(89)
        self.assertNotEqual(gain_relative_to_84, gain_relative_to_89)

    def test_r128_targetlevel_has_effect(self):
        if False:
            print('Hello World!')
        if not self.has_r128_support:
            self.skipTest('r128 tags for opus not supported on backend {}'.format(self.backend))
        album = self._add_album(1, ext='opus')
        item = album.items()[0]

        def analyse(target_level):
            if False:
                return 10
            self.config['replaygain']['r128_targetlevel'] = target_level
            self.run_command('replaygain', '-f')
            item.load()
            return item.r128_track_gain
        gain_relative_to_84 = analyse(84)
        gain_relative_to_89 = analyse(89)
        self.assertNotEqual(gain_relative_to_84, gain_relative_to_89)

    def test_per_disc(self):
        if False:
            print('Hello World!')
        album = self._add_album(track_count=4, disc_count=3)
        self.config['replaygain']['per_disc'] = True
        self.run_command('replaygain', '-a')
        for item in album.items():
            self.assertIsNotNone(item.rg_track_gain)
            self.assertIsNotNone(item.rg_album_gain)

@unittest.skipIf(not GST_AVAILABLE, 'gstreamer cannot be found')
class ReplayGainGstCliTest(ReplayGainCliTestBase, unittest.TestCase, GstBackendMixin):
    FNAME = 'full'

@unittest.skipIf(not GAIN_PROG_AVAILABLE, 'no *gain command found')
class ReplayGainCmdCliTest(ReplayGainCliTestBase, unittest.TestCase, CmdBackendMixin):
    FNAME = 'full'

@unittest.skipIf(not FFMPEG_AVAILABLE, 'ffmpeg cannot be found')
class ReplayGainFfmpegCliTest(ReplayGainCliTestBase, unittest.TestCase, FfmpegBackendMixin):
    FNAME = 'full'

@unittest.skipIf(not FFMPEG_AVAILABLE, 'ffmpeg cannot be found')
class ReplayGainFfmpegNoiseCliTest(ReplayGainCliTestBase, unittest.TestCase, FfmpegBackendMixin):
    FNAME = 'whitenoise'

class ImportTest(TestHelper):
    threaded = False

    def setUp(self):
        if False:
            print('Hello World!')
        self.test_backend()
        self.setup_beets(disk=True)
        self.config['threaded'] = self.threaded
        self.config['replaygain']['backend'] = self.backend
        try:
            self.load_plugins('replaygain')
        except Exception:
            self.teardown_beets()
            self.unload_plugins()
        self.importer = self.create_importer()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.unload_plugins()
        self.teardown_beets()

    def test_import_converted(self):
        if False:
            for i in range(10):
                print('nop')
        self.importer.run()
        for item in self.lib.items():
            self.assertIsNotNone(item.rg_track_gain)
            self.assertIsNotNone(item.rg_album_gain)

@unittest.skipIf(not GST_AVAILABLE, 'gstreamer cannot be found')
class ReplayGainGstImportTest(ImportTest, unittest.TestCase, GstBackendMixin):
    pass

@unittest.skipIf(not GAIN_PROG_AVAILABLE, 'no *gain command found')
class ReplayGainCmdImportTest(ImportTest, unittest.TestCase, CmdBackendMixin):
    pass

@unittest.skipIf(not FFMPEG_AVAILABLE, 'ffmpeg cannot be found')
class ReplayGainFfmpegImportTest(ImportTest, unittest.TestCase, FfmpegBackendMixin):
    pass

@unittest.skipIf(not FFMPEG_AVAILABLE, 'ffmpeg cannot be found')
class ReplayGainFfmpegThreadedImportTest(ImportTest, unittest.TestCase, FfmpegBackendMixin):
    threaded = True

def suite():
    if False:
        print('Hello World!')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')