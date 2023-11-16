"""
Test issue #336: Unicode problem w/ exception traceback on Windows (python2.7)
Default encoding (unicode-escape) of text() causes problems w/
exception tracebacks.

STATUS: BASICS SOLVED (default encoding changed)

ALTERNATIVE SOLUTIONS:
* Use traceback2: Returns unicode-string when calling traceback.
* Use text(traceback.format_exc(), sys.getfilesystemencoding(), "replace")
  where the text-conversion of a traceback is used.
  MAYBE traceback_to_text(traceback.format_exc())
"""
from __future__ import print_function
from behave.textutil import text
import pytest
import six

class TestIssue(object):
    traceback_bytes = b'\\\nTraceback (most recent call last):\n  File "C:\\Users\\alice\\xxx\\behave\\model.py", line 1456, in run\n    match.run(runner.context)\n  File "C:\\Users\\alice\\xxx\\behave\\model.py", line 1903, in run\n    self.func(context, args, *kwargs)\n  File "features\\steps\\my_steps.py", line 210, in step_impl\n    directories, task_names, reg_keys)\nAssertionError\n'
    traceback_file_line_texts = [u'File "C:\\Users\\alice\\xxx\\behave\\model.py", line 1456, in run', u'File "C:\\Users\\alice\\xxx\\behave\\model.py", line 1903, in run', u'File "features\\steps\\my_steps.py", line 210, in step_impl']

    def test_issue__with_default_encoding(self):
        if False:
            i = 10
            return i + 15
        'Test ensures that problem is fixed with default encoding'
        text2 = text(self.traceback_bytes)
        assert isinstance(self.traceback_bytes, bytes)
        assert isinstance(text2, six.text_type)
        for file_line_text in self.traceback_file_line_texts:
            assert file_line_text in text2

    @pytest.mark.filterwarnings('ignore:invalid escape sequence')
    def test__problem_exists_with_problematic_encoding(self):
        if False:
            while True:
                i = 10
        'Test ensures that problem exists with encoding=unicode-escape'
        problematic_encoding = 'unicode-escape'
        text2 = text(self.traceback_bytes, problematic_encoding)
        print('TEXT: ' + text2)
        assert isinstance(self.traceback_bytes, bytes)
        assert isinstance(text2, six.text_type)
        file_line_text = self.traceback_file_line_texts[0]
        assert file_line_text not in text2