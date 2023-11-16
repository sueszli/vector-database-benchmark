from __future__ import annotations
import collections
from unittest.mock import Mock
import unittest
from ansible.modules.apt import expand_pkgspec_from_fnmatches

class AptExpandPkgspecTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        FakePackage = collections.namedtuple('Package', ('name',))
        self.fake_cache = [FakePackage('apt'), FakePackage('apt-utils'), FakePackage('not-selected')]

    def test_trivial(self):
        if False:
            return 10
        pkg = ['apt']
        self.assertEqual(expand_pkgspec_from_fnmatches(None, pkg, self.fake_cache), pkg)

    def test_version_wildcard(self):
        if False:
            for i in range(10):
                print('nop')
        pkg = ['apt=1.0*']
        self.assertEqual(expand_pkgspec_from_fnmatches(None, pkg, self.fake_cache), pkg)

    def test_pkgname_wildcard_version_wildcard(self):
        if False:
            i = 10
            return i + 15
        pkg = ['apt*=1.0*']
        m_mock = Mock()
        self.assertEqual(expand_pkgspec_from_fnmatches(m_mock, pkg, self.fake_cache), ['apt', 'apt-utils'])

    def test_pkgname_expands(self):
        if False:
            while True:
                i = 10
        pkg = ['apt*']
        m_mock = Mock()
        self.assertEqual(expand_pkgspec_from_fnmatches(m_mock, pkg, self.fake_cache), ['apt', 'apt-utils'])