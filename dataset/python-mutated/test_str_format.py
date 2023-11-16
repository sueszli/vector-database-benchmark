from __future__ import annotations
import unittest
from test_case_base import TestCaseBase

def find_spec(self, fullname, path, target=None):
    if False:
        return 10
    method_name = 'spec_for_{fullname}'.format(**{'self': self, 'fullname': fullname})
    method = getattr(self, method_name, lambda : None)
    return method()

class TestExecutor(TestCaseBase):

    def test_simple(self):
        if False:
            print('Hello World!')
        self.assert_results(find_spec, 'self', 'fullname', 'path', None)
if __name__ == '__main__':
    unittest.main()