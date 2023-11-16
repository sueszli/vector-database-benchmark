"""Test functions with complex cfg."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TestComplexFunction(test_base.BaseTest):
    """Test function with complex cfg."""

    def test_function_not_optimized(self):
        if False:
            return 10
        code = test_utils.test_data_file('tokenize.py')
        with self.DepTree([('foo.py', code)]):
            self.Check('\n        import io\n        import foo\n        stream = io.StringIO("")\n        tokens = foo.generate_tokens(stream.readline)\n        for tok_type, tok_str, _, _, _ in tokens:\n          pass\n      ')
if __name__ == '__main__':
    test_base.main()