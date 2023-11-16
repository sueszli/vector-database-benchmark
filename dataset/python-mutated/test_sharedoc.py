from zipline.testing import ZiplineTestCase
from zipline.utils.sharedoc import copydoc

class TestSharedoc(ZiplineTestCase):

    def test_copydoc(self):
        if False:
            return 10

        def original_docstring_function():
            if False:
                print('Hello World!')
            '\n            My docstring brings the boys to the yard.\n            '
            pass

        @copydoc(original_docstring_function)
        def copied_docstring_function():
            if False:
                return 10
            pass
        self.assertEqual(original_docstring_function.__doc__, copied_docstring_function.__doc__)