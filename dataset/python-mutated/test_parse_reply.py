import unittest
from trashcli.empty.parse_reply import parse_reply

class TestParseReply(unittest.TestCase):

    def test_y(self):
        if False:
            for i in range(10):
                print('nop')
        assert parse_reply('y') == True

    def test_Y(self):
        if False:
            return 10
        assert parse_reply('Y') == True

    def test_n(self):
        if False:
            i = 10
            return i + 15
        assert parse_reply('n') == False

    def test_N(self):
        if False:
            for i in range(10):
                print('nop')
        assert parse_reply('N') == False

    def test_empty_string(self):
        if False:
            for i in range(10):
                print('nop')
        assert parse_reply('') == False