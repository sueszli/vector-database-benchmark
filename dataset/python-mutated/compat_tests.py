import unittest
from pika import compat

class UtilsTests(unittest.TestCase):

    def test_get_linux_version_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(compat.get_linux_version('4.11.0-2-amd64'), (4, 11, 0))

    def test_get_linux_version_short(self):
        if False:
            return 10
        self.assertEqual(compat.get_linux_version('4.11.0'), (4, 11, 0))

    def test_get_linux_version_gcp(self):
        if False:
            while True:
                i = 10
        self.assertEqual(compat.get_linux_version('4.4.64+'), (4, 4, 64))

    def test_to_digit(self):
        if False:
            print('Hello World!')
        self.assertEqual(compat.to_digit('64'), 64)

    def test_to_digit_with_plus_sign(self):
        if False:
            print('Hello World!')
        self.assertEqual(compat.to_digit('64+'), 64)

    def test_to_digit_with_dot(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(compat.to_digit('64.'), 64)