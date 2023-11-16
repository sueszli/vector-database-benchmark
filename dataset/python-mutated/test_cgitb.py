from test.support.os_helper import temp_dir
from test.support.script_helper import assert_python_failure
import unittest
import sys
import cgitb

class TestCgitb(unittest.TestCase):

    def test_fonts(self):
        if False:
            i = 10
            return i + 15
        text = 'Hello Robbie!'
        self.assertEqual(cgitb.small(text), '<small>{}</small>'.format(text))
        self.assertEqual(cgitb.strong(text), '<strong>{}</strong>'.format(text))
        self.assertEqual(cgitb.grey(text), '<font color="#909090">{}</font>'.format(text))

    def test_blanks(self):
        if False:
            return 10
        self.assertEqual(cgitb.small(''), '')
        self.assertEqual(cgitb.strong(''), '')
        self.assertEqual(cgitb.grey(''), '')

    def test_html(self):
        if False:
            while True:
                i = 10
        try:
            raise ValueError('Hello World')
        except ValueError as err:
            html = cgitb.html(sys.exc_info())
            self.assertIn('ValueError', html)
            self.assertIn(str(err), html)

    def test_text(self):
        if False:
            while True:
                i = 10
        try:
            raise ValueError('Hello World')
        except ValueError:
            text = cgitb.text(sys.exc_info())
            self.assertIn('ValueError', text)
            self.assertIn('Hello World', text)

    def test_syshook_no_logdir_default_format(self):
        if False:
            while True:
                i = 10
        with temp_dir() as tracedir:
            (rc, out, err) = assert_python_failure('-c', 'import cgitb; cgitb.enable(logdir=%s); raise ValueError("Hello World")' % repr(tracedir), PYTHONIOENCODING='utf-8')
        out = out.decode()
        self.assertIn('ValueError', out)
        self.assertIn('Hello World', out)
        self.assertIn('<strong>&lt;module&gt;</strong>', out)
        self.assertIn('<p>', out)
        self.assertIn('</p>', out)

    def test_syshook_no_logdir_text_format(self):
        if False:
            i = 10
            return i + 15
        with temp_dir() as tracedir:
            (rc, out, err) = assert_python_failure('-c', 'import cgitb; cgitb.enable(format="text", logdir=%s); raise ValueError("Hello World")' % repr(tracedir), PYTHONIOENCODING='utf-8')
        out = out.decode()
        self.assertIn('ValueError', out)
        self.assertIn('Hello World', out)
        self.assertNotIn('<p>', out)
        self.assertNotIn('</p>', out)
if __name__ == '__main__':
    unittest.main()