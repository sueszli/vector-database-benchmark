"""Keyword args tests."""
from tensorflow.python.platform import test
from tensorflow.python.util import keyword_args

class KeywordArgsTest(test.TestCase):

    def test_keyword_args_only(self):
        if False:
            i = 10
            return i + 15

        def func_without_decorator(a, b):
            if False:
                print('Hello World!')
            return a + b

        @keyword_args.keyword_args_only
        def func_with_decorator(a, b):
            if False:
                while True:
                    i = 10
            return func_without_decorator(a, b)
        self.assertEqual(3, func_without_decorator(1, 2))
        self.assertEqual(3, func_without_decorator(a=1, b=2))
        self.assertEqual(3, func_with_decorator(a=1, b=2))
        with self.assertRaisesRegex(ValueError, 'only accepts keyword arguments'):
            self.assertEqual(3, func_with_decorator(1, 2))
        with self.assertRaisesRegex(ValueError, 'only accepts keyword arguments'):
            self.assertEqual(3, func_with_decorator(1, b=2))
if __name__ == '__main__':
    test.main()