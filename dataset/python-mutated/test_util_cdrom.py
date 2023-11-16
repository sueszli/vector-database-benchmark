import io
from typing import Iterable
import unittest
from test.picardtestcase import PicardTestCase
from picard.const.sys import IS_WIN
from picard.util import cdrom
MOCK_CDROM_INFO = 'CD-ROM information, Id: cdrom.c 3.20 2003/12/17\n\ndrive name:\t\tsr1\tsr0\ndrive speed:\t\t24\t24\ndrive # of slots:\t1\t1\nCan close tray:\t\t0\t1\nCan open tray:\t\t1\t1\nCan lock tray:\t\t1\t1\nCan change speed:\t1\t1\nCan select disk:\t0\t0\nCan read multisession:\t1\t1\nCan read MCN:\t\t1\t1\nReports media changed:\t1\t1\nCan play audio:\t\t1\t0\nCan write CD-R:\t\t1\t1\nCan write CD-RW:\t1\t1\nCan read DVD:\t\t1\t1\nCan write DVD-R:\t1\t1\nCan write DVD-RAM:\t1\t1\nCan read MRW:\t\t1\t1\nCan write MRW:\t\t1\t1\nCan write RAM:\t\t1\t1\n'
MOCK_CDROM_INFO_EMPTY = 'CD-ROM information, Id: cdrom.c 3.20 2003/12/17\n\ndrive name:\ndrive speed:\ndrive # of slots:\nCan close tray:\nCan open tray:\nCan lock tray:\nCan change speed:\nCan select disk:\nCan read multisession:\nCan read MCN:\nReports media changed:\nCan play audio:\nCan write CD-R:\nCan write CD-RW:\nCan read DVD:\nCan write DVD-R:\nCan write DVD-RAM:\nCan read MRW:\nCan write MRW:\nCan write RAM:\n'

class LinuxParseCdromInfoTest(PicardTestCase):

    def test_drives(self):
        if False:
            print('Hello World!')
        with io.StringIO(MOCK_CDROM_INFO) as f:
            drives = list(cdrom._parse_linux_cdrom_info(f))
            self.assertEqual([('sr1', True), ('sr0', False)], drives)

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        with io.StringIO(MOCK_CDROM_INFO_EMPTY) as f:
            drives = list(cdrom._parse_linux_cdrom_info(f))
            self.assertEqual([], drives)

    def test_empty_string(self):
        if False:
            print('Hello World!')
        with io.StringIO('') as f:
            drives = list(cdrom._parse_linux_cdrom_info(f))
            self.assertEqual([], drives)

class GetCdromDrivesTest(PicardTestCase):

    def test_get_cdrom_drives(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_config_values({'cd_lookup_device': '/dev/cdrom'})
        drives = cdrom.get_cdrom_drives()
        self.assertIsInstance(drives, Iterable)
        self.assertTrue(set(cdrom.DEFAULT_DRIVES).issubset(drives))

    def test_generic_iter_drives(self):
        if False:
            print('Hello World!')
        self.set_config_values({'cd_lookup_device': '/dev/cdrom'})
        self.assertEqual(['/dev/cdrom'], list(cdrom._generic_iter_drives()))
        self.set_config_values({'cd_lookup_device': '/dev/cdrom, /dev/sr0'})
        self.assertEqual(['/dev/cdrom', '/dev/sr0'], list(cdrom._generic_iter_drives()))
        self.set_config_values({'cd_lookup_device': ''})
        self.assertEqual([], list(cdrom._generic_iter_drives()))
        self.set_config_values({'cd_lookup_device': ' ,, ,\t, '})
        self.assertEqual([], list(cdrom._generic_iter_drives()))

@unittest.skipUnless(IS_WIN, 'windows test')
class WindowsGetCdromDrivesTest(PicardTestCase):

    def test_autodetect(self):
        if False:
            print('Hello World!')
        self.assertTrue(cdrom.AUTO_DETECT_DRIVES)

    def test_iter_drives(self):
        if False:
            i = 10
            return i + 15
        drives = cdrom._iter_drives()
        self.assertIsInstance(drives, Iterable)
        list(drives)