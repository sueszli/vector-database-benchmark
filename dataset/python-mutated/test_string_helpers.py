__license__ = 'GNU Affero General Public License http://www.gnu.org/licenses/agpl.html'
__copyright__ = 'Copyright (C) 2019 The OctoPrint Project - Released under terms of the AGPLv3 License'
import unittest
import pytest
import octoprint.util

class StringHelperTest(unittest.TestCase):

    def test_to_unicode_unicode(self):
        if False:
            print('Hello World!')
        result = octoprint.util.to_unicode('test')
        self.assertEqual(result, 'test')
        self.assertIsInstance(result, str)

    def test_to_unicode_bytes(self):
        if False:
            print('Hello World!')
        result = octoprint.util.to_unicode(b'test')
        self.assertEqual(result, 'test')
        self.assertIsInstance(result, str)

    def test_to_unicode_bytes_utf8(self):
        if False:
            i = 10
            return i + 15
        data = 'äöüß'
        result = octoprint.util.to_unicode(data.encode('utf-8'), encoding='utf-8')
        self.assertEqual(result, data)
        self.assertIsInstance(result, str)

    def test_to_unicode_bytes_utf8_vs_ascii(self):
        if False:
            return 10
        self.assertRaises(UnicodeDecodeError, octoprint.util.to_unicode, 'äöüß'.encode(), encoding='ascii')

    def test_to_unicode_bytes_utf8_vs_ascii_replace(self):
        if False:
            return 10
        data = 'äöüß'
        result = octoprint.util.to_unicode(data.encode('utf-8'), encoding='ascii', errors='replace')
        self.assertEqual(result, data.encode('utf-8').decode('ascii', errors='replace'))
        self.assertIsInstance(result, str)

    def test_to_bytes_bytes(self):
        if False:
            i = 10
            return i + 15
        result = octoprint.util.to_bytes(b'test')
        self.assertEqual(result, b'test')
        self.assertIsInstance(result, bytes)

    def test_to_bytes_str(self):
        if False:
            print('Hello World!')
        result = octoprint.util.to_bytes('test')
        self.assertEqual(result, b'test')
        self.assertIsInstance(result, bytes)

    def test_to_bytes_str_utf8(self):
        if False:
            i = 10
            return i + 15
        data = 'äöüß'
        result = octoprint.util.to_bytes(data, encoding='utf-8')
        self.assertEqual(result, data.encode('utf-8'))
        self.assertIsInstance(result, bytes)

    def test_to_bytes_str_utf8_vs_ascii(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(UnicodeEncodeError, octoprint.util.to_bytes, 'äöüß', encoding='ascii')

    def test_to_bytes_str_utf8_vs_ascii_replace(self):
        if False:
            i = 10
            return i + 15
        data = 'äöüß'
        result = octoprint.util.to_bytes(data, encoding='ascii', errors='replace')
        self.assertEqual(result, data.encode('ascii', errors='replace'))
        self.assertIsInstance(result, bytes)

    def test_to_str(self):
        if False:
            return 10
        with pytest.deprecated_call():
            result = octoprint.util.to_str('test')
            self.assertEqual(result, b'test')
            self.assertIsInstance(result, bytes)

    def test_to_native_str(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.deprecated_call():
            result = octoprint.util.to_native_str(b'test')
            self.assertEqual(result, 'test')
            self.assertIsInstance(result, str)