import os
import unittest

class TestWinutil(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        from calibre_extensions import winutil
        self.winutil = winutil

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        del self.winutil

    def test_add_to_recent_docs(self):
        if False:
            for i in range(10):
                print('nop')
        path = str(os.path.abspath(__file__))
        self.winutil.add_to_recent_docs(path, None)
        self.winutil.add_to_recent_docs(path, 'some-app-uid')

    def test_file_association(self):
        if False:
            i = 10
            return i + 15
        q = self.winutil.file_association('.txt')
        self.assertIn('notepad.exe', q.lower())
        self.assertNotIn('\x00', q)
        q = self.winutil.friendly_name(None, 'notepad.exe')
        self.assertEqual('Notepad', q)

    def test_special_folder_path(self):
        if False:
            print('Hello World!')
        self.assertEqual(os.path.expanduser('~'), self.winutil.special_folder_path(self.winutil.CSIDL_PROFILE))

    def test_associations_changed(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(self.winutil.notify_associations_changed())

def find_tests():
    if False:
        return 10
    return unittest.defaultTestLoader.loadTestsFromTestCase(TestWinutil)