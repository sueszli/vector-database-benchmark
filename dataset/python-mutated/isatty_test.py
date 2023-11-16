import sys
from unittest import TestCase, main
from ..ansitowin32 import StreamWrapper, AnsiToWin32
from .utils import pycharm, replace_by, replace_original_by, StreamTTY, StreamNonTTY

def is_a_tty(stream):
    if False:
        for i in range(10):
            print('nop')
    return StreamWrapper(stream, None).isatty()

class IsattyTest(TestCase):

    def test_TTY(self):
        if False:
            i = 10
            return i + 15
        tty = StreamTTY()
        self.assertTrue(is_a_tty(tty))
        with pycharm():
            self.assertTrue(is_a_tty(tty))

    def test_nonTTY(self):
        if False:
            while True:
                i = 10
        non_tty = StreamNonTTY()
        self.assertFalse(is_a_tty(non_tty))
        with pycharm():
            self.assertFalse(is_a_tty(non_tty))

    def test_withPycharm(self):
        if False:
            for i in range(10):
                print('nop')
        with pycharm():
            self.assertTrue(is_a_tty(sys.stderr))
            self.assertTrue(is_a_tty(sys.stdout))

    def test_withPycharmTTYOverride(self):
        if False:
            while True:
                i = 10
        tty = StreamTTY()
        with pycharm(), replace_by(tty):
            self.assertTrue(is_a_tty(tty))

    def test_withPycharmNonTTYOverride(self):
        if False:
            print('Hello World!')
        non_tty = StreamNonTTY()
        with pycharm(), replace_by(non_tty):
            self.assertFalse(is_a_tty(non_tty))

    def test_withPycharmNoneOverride(self):
        if False:
            while True:
                i = 10
        with pycharm():
            with replace_by(None), replace_original_by(None):
                self.assertFalse(is_a_tty(None))
                self.assertFalse(is_a_tty(StreamNonTTY()))
                self.assertTrue(is_a_tty(StreamTTY()))

    def test_withPycharmStreamWrapped(self):
        if False:
            for i in range(10):
                print('nop')
        with pycharm():
            self.assertTrue(AnsiToWin32(StreamTTY()).stream.isatty())
            self.assertFalse(AnsiToWin32(StreamNonTTY()).stream.isatty())
            self.assertTrue(AnsiToWin32(sys.stdout).stream.isatty())
            self.assertTrue(AnsiToWin32(sys.stderr).stream.isatty())
if __name__ == '__main__':
    main()