"""This file contains test cases reported by third parties using
fuzzing tools, primarily from Google's oss-fuzz project. Some of these
represent real problems with Beautiful Soup, but many are problems in
libraries that Beautiful Soup depends on, and many of the test cases
represent different ways of triggering the same problem.

Grouping these test cases together makes it easy to see which test
cases represent the same problem, and puts the test cases in close
proximity to code that can trigger the problems.
"""
import os
import pytest
from bs4 import BeautifulSoup, ParserRejectedMarkup

class TestFuzz(object):
    TESTCASE_SUFFIX = '.testcase'

    @pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-5703933063462912'])
    def test_rejected_markup(self, filename):
        if False:
            while True:
                i = 10
        markup = self.__markup(filename)
        with pytest.raises(ParserRejectedMarkup):
            BeautifulSoup(markup, 'html.parser')

    @pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-5984173902397440', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5167584867909632', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6124268085182464', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6450958476902400'])
    def test_deeply_nested_document(self, filename):
        if False:
            print('Hello World!')
        markup = self.__markup(filename)
        BeautifulSoup(markup, 'html.parser').encode()

    @pytest.mark.skip('html5lib problems')
    @pytest.mark.parametrize('filename', ['clusterfuzz-testcase-minimized-bs4_fuzzer-4818336571064320', 'clusterfuzz-testcase-minimized-bs4_fuzzer-4999465949331456', 'clusterfuzz-testcase-minimized-bs4_fuzzer-5843991618256896', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6241471367348224', 'clusterfuzz-testcase-minimized-bs4_fuzzer-6600557255327744', 'crash-0d306a50c8ed8bcd0785b67000fcd5dea1d33f08'])
    def test_html5lib_parse_errors(self, filename):
        if False:
            return 10
        markup = self.__markup(filename)
        print(BeautifulSoup(markup, 'html5lib').encode())

    def __markup(self, filename):
        if False:
            i = 10
            return i + 15
        if not filename.endswith(self.TESTCASE_SUFFIX):
            filename += self.TESTCASE_SUFFIX
        this_dir = os.path.split(__file__)[0]
        path = os.path.join(this_dir, 'fuzz', filename)
        return open(path, 'rb').read()