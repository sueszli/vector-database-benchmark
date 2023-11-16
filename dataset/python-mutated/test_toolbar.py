from __future__ import print_function
from tests.compat import unittest
from pygments.token import Token
from haxor_news.haxor import Haxor
from haxor_news.toolbar import Toolbar

class ToolbarTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.haxor = Haxor()
        self.toolbar = Toolbar(lambda : self.haxor.paginate_comments)

    def test_toolbar_on(self):
        if False:
            while True:
                i = 10
        self.haxor.paginate_comments = True
        expected = [(Token.Toolbar, ' [F10] Exit ')]
        assert expected == self.toolbar.handler(None)

    def test_toolbar_off(self):
        if False:
            for i in range(10):
                print('nop')
        self.haxor.paginate_comments = False
        expected = [(Token.Toolbar, ' [F10] Exit ')]
        assert expected == self.toolbar.handler(None)