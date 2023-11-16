import unittest
from test.helper import TestHelper, capture_stdout, control_stdin
from test.test_importer import AutotagStub, ImportHelper
from test.test_ui_importer import TerminalImportSessionSetup

class MBSubmitPluginTest(TerminalImportSessionSetup, unittest.TestCase, ImportHelper, TestHelper):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_beets()
        self.load_plugins('mbsubmit')
        self._create_import_dir(2)
        self._setup_import_session()
        self.matcher = AutotagStub().install()

    def tearDown(self):
        if False:
            return 10
        self.unload_plugins()
        self.teardown_beets()
        self.matcher.restore()

    def test_print_tracks_output(self):
        if False:
            while True:
                i = 10
        'Test the output of the "print tracks" choice.'
        self.matcher.matching = AutotagStub.BAD
        with capture_stdout() as output:
            with control_stdin('\n'.join(['p', 's'])):
                self.importer.run()
        tracklist = 'Print tracks? 01. Tag Title 1 - Tag Artist (0:01)\n02. Tag Title 2 - Tag Artist (0:01)'
        self.assertIn(tracklist, output.getvalue())

    def test_print_tracks_output_as_tracks(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the output of the "print tracks" choice, as singletons.'
        self.matcher.matching = AutotagStub.BAD
        with capture_stdout() as output:
            with control_stdin('\n'.join(['t', 's', 'p', 's'])):
                self.importer.run()
        tracklist = 'Print tracks? 02. Tag Title 2 - Tag Artist (0:01)'
        self.assertIn(tracklist, output.getvalue())

def suite():
    if False:
        print('Hello World!')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')