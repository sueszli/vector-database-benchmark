from __future__ import absolute_import
import unittest2
from st2common.util.versioning import complex_semver_match
from st2common.util.pack import normalize_pack_version

class VersioningUtilsTestCase(unittest2.TestCase):

    def test_complex_semver_match(self):
        if False:
            while True:
                i = 10
        self.assertTrue(complex_semver_match('1.6.0', '>=1.6.0, <2.2.0'))
        self.assertTrue(complex_semver_match('1.6.1', '>=1.6.0, <2.2.0'))
        self.assertTrue(complex_semver_match('2.0.0', '>=1.6.0, <2.2.0'))
        self.assertTrue(complex_semver_match('2.1.0', '>=1.6.0, <2.2.0'))
        self.assertTrue(complex_semver_match('2.1.9', '>=1.6.0, <2.2.0'))
        self.assertTrue(complex_semver_match('1.6.0', 'all'))
        self.assertTrue(complex_semver_match('1.6.1', 'all'))
        self.assertTrue(complex_semver_match('2.0.0', 'all'))
        self.assertTrue(complex_semver_match('2.1.0', 'all'))
        self.assertTrue(complex_semver_match('1.6.0', '>=1.6.0'))
        self.assertTrue(complex_semver_match('1.6.1', '>=1.6.0'))
        self.assertTrue(complex_semver_match('2.1.0', '>=1.6.0'))
        self.assertFalse(complex_semver_match('1.5.0', '>=1.6.0, <2.2.0'))
        self.assertFalse(complex_semver_match('0.1.0', '>=1.6.0, <2.2.0'))
        self.assertFalse(complex_semver_match('2.2.1', '>=1.6.0, <2.2.0'))
        self.assertFalse(complex_semver_match('2.3.0', '>=1.6.0, <2.2.0'))
        self.assertFalse(complex_semver_match('3.0.0', '>=1.6.0, <2.2.0'))
        self.assertFalse(complex_semver_match('1.5.0', '>=1.6.0'))
        self.assertFalse(complex_semver_match('0.1.0', '>=1.6.0'))
        self.assertFalse(complex_semver_match('1.5.9', '>=1.6.0'))

    def test_normalize_pack_version(self):
        if False:
            print('Hello World!')
        self.assertEqual(normalize_pack_version('0.2.0'), '0.2.0')
        self.assertEqual(normalize_pack_version('0.2.1'), '0.2.1')
        self.assertEqual(normalize_pack_version('1.2.1'), '1.2.1')
        self.assertEqual(normalize_pack_version('0.2'), '0.2.0')
        self.assertEqual(normalize_pack_version('0.3'), '0.3.0')
        self.assertEqual(normalize_pack_version('1.3'), '1.3.0')
        self.assertEqual(normalize_pack_version('2.0'), '2.0.0')