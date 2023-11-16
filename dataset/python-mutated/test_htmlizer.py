"""
Tests for L{twisted.python.htmlizer}.
"""
from io import BytesIO
from twisted.python.htmlizer import filter
from twisted.trial.unittest import TestCase

class FilterTests(TestCase):
    """
    Tests for L{twisted.python.htmlizer.filter}.
    """

    def test_empty(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        If passed an empty input file, L{filter} writes a I{pre} tag containing\n        only an end marker to the output file.\n        '
        input = BytesIO(b'')
        output = BytesIO()
        filter(input, output)
        self.assertEqual(output.getvalue(), b'<pre><span class="py-src-endmarker"></span></pre>\n')

    def test_variable(self) -> None:
        if False:
            return 10
        '\n        If passed an input file containing a variable access, L{filter} writes\n        a I{pre} tag containing a I{py-src-variable} span containing the\n        variable.\n        '
        input = BytesIO(b'foo\n')
        output = BytesIO()
        filter(input, output)
        self.assertEqual(output.getvalue(), b'<pre><span class="py-src-variable">foo</span><span class="py-src-newline">\n</span><span class="py-src-endmarker"></span></pre>\n')