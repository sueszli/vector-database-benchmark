import unittest
import pytest
from kivy import platform
unicode_char = chr

class FileChooserUnicodeTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.skip_test = platform == 'macosx' or platform == 'ios'
        if self.skip_test:
            return
        import os
        from os.path import join
        from zipfile import ZipFile
        basepath = os.path.dirname(__file__) + u''
        basepathu = join(basepath, u'filechooser_files')
        self.basepathu = basepathu
        basepathb = os.path.dirname(__file__.encode())
        basepathb = join(basepathb, b'filechooser_files')
        self.assertIsInstance(basepathb, bytes)
        self.basepathb = basepathb
        ufiles = [u'कीवीtestu', u'कीवीtestu' + unicode_char(61166), u'कीवीtestu' + unicode_char(61166 - 1), u'कीवीtestu' + unicode_char(238)]
        bfiles = [b'\xc3\xa0\xc2\xa4\xe2\x80\xa2\xc3\xa0\xc2\xa5\xe2\x82\xac        \xc3\xa0\xc2\xa4\xc2\xb5\xc3\xa0\xc2\xa5\xe2\x82\xactestb', b'oor\xff\xff\xff\xff\xee\xfe\xef\x81\x8d\x99testb']
        self.ufiles = [join(basepathu, f) for f in ufiles]
        self.bfiles = []
        if not os.path.isdir(basepathu):
            os.mkdir(basepathu)
        for f in self.ufiles:
            open(f, 'wb').close()
        for f in self.bfiles:
            open(f, 'wb').close()
        existfiles = [u'à¤•à¥€à¤µà¥€test', u'à¤•à¥€à¤’µà¥€test', u'Ã\xa0Â¤â€¢Ã\xa0Â¥â‚¬Ã\xa0Â¤ÂµÃ\xa0Â¥â‚¬test', u'testl\ufffe', u'testl\uffff']
        self.exitsfiles = [join(basepathu, f) for f in existfiles]
        with ZipFile(join(basepath, u'unicode_files.zip'), 'r') as myzip:
            myzip.extractall(path=basepathu)
        for f in self.exitsfiles:
            open(f, 'rb').close()

    @pytest.fixture(autouse=True)
    def set_clock(self, kivy_clock):
        if False:
            return 10
        self.kivy_clock = kivy_clock

    def test_filechooserlistview_unicode(self):
        if False:
            while True:
                i = 10
        if self.skip_test:
            return
        from kivy.uix.filechooser import FileChooserListView
        from kivy.clock import Clock
        from os.path import join
        wid = FileChooserListView(path=self.basepathu)
        for i in range(1):
            Clock.tick()
        files = [join(self.basepathu, f) for f in wid.files]
        for f in self.ufiles:
            self.assertIn(f, files)
        for f in self.exitsfiles:
            self.assertIn(f, files)

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self.skip_test:
            return
        from os import remove, rmdir
        try:
            for f in self.ufiles:
                remove(f)
            for f in self.exitsfiles:
                remove(f)
            for f in self.bfiles:
                remove(f)
            rmdir(self.basepathu)
        except:
            pass