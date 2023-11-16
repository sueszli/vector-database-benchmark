import unittest
from coalib.bearlib.aspects.exceptions import AspectLookupError

class ExceptionTest(unittest.TestCase):

    def test_AspectLookupError(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(AspectLookupError, "^Error when trying to search aspect named 'NOASPECT'$"):
            raise AspectLookupError('NOASPECT')