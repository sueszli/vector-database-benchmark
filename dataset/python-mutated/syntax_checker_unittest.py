"""
WIP

Yes, we do write tests for our tests.
"""
from pathlib import Path
from typing import List
from unittest import TestCase
from tests import syntax_lint

def open_test_case_file(n: int) -> List[bytes]:
    if False:
        for i in range(10):
            print('nop')
    tests_path = Path(__file__).parent.joinpath()
    with open(f'{tests_path}/testcases/testcase{n}.md', 'rb') as f:
        file_list = [line.rstrip() for line in f.readlines()]
    return file_list
test_case_1 = open_test_case_file(1)
test_case_2 = open_test_case_file(2)
test_case_3 = open_test_case_file(3)

class TestSyntax(TestCase):

    def test_details_count_case1(self):
        if False:
            while True:
                i = 10
        self.assertTrue(syntax_lint.count_details(test_case_1))

    def test_details_count_case2(self):
        if False:
            while True:
                i = 10
        self.assertTrue(syntax_lint.count_details(test_case_2))

    def test_details_errors_1(self):
        if False:
            for i in range(10):
                print('nop')
        syntax_lint.check_details_tag(test_case_1)
        self.assertFalse(syntax_lint.errors)

    def test_details_errors_2(self):
        if False:
            i = 10
            return i + 15
        syntax_lint.check_details_tag(test_case_2)
        self.assertFalse(syntax_lint.errors)