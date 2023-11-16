import unittest
from shutil import rmtree
from os.path import exists, isfile, join
from tempfile import gettempdir
from mycroft.util import create_signal, check_for_signal

class TestSignals(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        if exists(join(gettempdir(), 'mycroft')):
            rmtree(join(gettempdir(), 'mycroft'))

    def test_create_signal(self):
        if False:
            return 10
        create_signal('test_signal')
        self.assertTrue(isfile(join(gettempdir(), 'mycroft/ipc/signal/test_signal')))

    def test_check_signal(self):
        if False:
            i = 10
            return i + 15
        if exists(join(gettempdir(), 'mycroft')):
            rmtree(join(gettempdir(), 'mycroft'))
        self.assertFalse(check_for_signal('test_signal'))
        create_signal('test_signal')
        self.assertTrue(check_for_signal('test_signal'))
        self.assertFalse(isfile(join(gettempdir(), 'mycroft/ipc/signal/test_signal')))
if __name__ == '__main__':
    unittest.main()