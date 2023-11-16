from io import StringIO
from unittest import TestCase
from pymysql.optionfile import Parser
__all__ = ['TestParser']
_cfg_file = '\n[default]\nstring = foo\nquoted = "bar"\nsingle_quoted = \'foobar\'\nskip-slave-start\n'

class TestParser(TestCase):

    def test_string(self):
        if False:
            for i in range(10):
                print('nop')
        parser = Parser()
        parser.read_file(StringIO(_cfg_file))
        self.assertEqual(parser.get('default', 'string'), 'foo')
        self.assertEqual(parser.get('default', 'quoted'), 'bar')
        self.assertEqual(parser.get('default', 'single-quoted'), 'foobar')