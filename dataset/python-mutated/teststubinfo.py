from __future__ import annotations
import unittest
from mypy.stubinfo import is_legacy_bundled_package

class TestStubInfo(unittest.TestCase):

    def test_is_legacy_bundled_packages(self) -> None:
        if False:
            while True:
                i = 10
        assert not is_legacy_bundled_package('foobar_asdf')
        assert is_legacy_bundled_package('pycurl')
        assert is_legacy_bundled_package('dataclasses')