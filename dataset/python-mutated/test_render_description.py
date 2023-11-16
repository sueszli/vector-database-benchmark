from twisted.trial import unittest
from buildbot.util.render_description import render_description

class TestRaml(unittest.TestCase):

    def test_plain(self):
        if False:
            return 10
        self.assertIsNone(render_description('description', None))

    def test_unknown(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(Exception):
            render_description('description', 'unknown')

    def test_markdown(self):
        if False:
            return 10
        self.assertEqual(render_description('# description\ntext', 'markdown'), '<h1>description</h1>\n<p>text</p>')