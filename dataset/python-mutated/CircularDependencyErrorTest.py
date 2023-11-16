import unittest
from coalib.core.CircularDependencyError import CircularDependencyError

class CircularDependencyErrorTest(unittest.TestCase):

    def test_default_message(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(CircularDependencyError) as cm:
            raise CircularDependencyError
        self.assertEqual(str(cm.exception), 'Circular dependency detected.')

    def test_message_with_dependency_circle(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(CircularDependencyError) as cm:
            raise CircularDependencyError(['A', 'B', 'C'])
        self.assertEqual(str(cm.exception), 'Circular dependency detected: A -> B -> C')