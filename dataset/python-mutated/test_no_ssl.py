"""
Test what happens if Python was built without SSL

* Everything that does not involve HTTPS should still work
* HTTPS requests must fail with an error that points at the ssl module
"""
from __future__ import annotations
import sys
from test import ImportBlocker, ModuleStash
import pytest
ssl_blocker = ImportBlocker('ssl', '_ssl')
module_stash = ModuleStash('urllib3')

class TestWithoutSSL:

    @classmethod
    def setup_class(cls) -> None:
        if False:
            print('Hello World!')
        sys.modules.pop('ssl', None)
        sys.modules.pop('_ssl', None)
        module_stash.stash()
        sys.meta_path.insert(0, ssl_blocker)

    @classmethod
    def teardown_class(cls) -> None:
        if False:
            return 10
        sys.meta_path.remove(ssl_blocker)
        module_stash.pop()

class TestImportWithoutSSL(TestWithoutSSL):

    def test_cannot_import_ssl(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.raises(ImportError):
            import ssl

    def test_import_urllib3(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        import urllib3