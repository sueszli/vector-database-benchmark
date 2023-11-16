import os
import unittest
from pydocstyle.checker import check, violations
from tests.utils import list_all_py_files
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
registry = violations.ErrorRegistry

def lookup_error_params(code):
    if False:
        for i in range(10):
            print('nop')
    for group in registry.groups:
        for error_params in group.errors:
            if error_params.code == code:
                return error_params

class PyDOC_Style_Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.violations = list()
        _disabled_checks = ['D205', 'D102', 'D400', 'D100', 'D107', 'D103', 'D401', 'D101', 'D413', 'D105', 'D104', 'D302', 'D202']
        for filename in list_all_py_files():
            print(filename)
            for err in check([filename]):
                if not err.code in _disabled_checks:
                    cls.violations.append(err)

    def test_violations(self):
        if False:
            print('Hello World!')
        if self.violations:
            counts = dict()
            for err in self.violations:
                counts[err.code] = counts.get(err.code, 0) + 1
                print(err)
            for (n, code) in sorted([(n, code) for (code, n) in counts.items()], reverse=True):
                p = lookup_error_params(code)
                print('%s %8d %s' % (code, n, p.short_desc))
            raise Exception('PyDoc Coding Style: %d violations have been found' % len(self.violations))
if __name__ == '__main__':
    unittest.main()