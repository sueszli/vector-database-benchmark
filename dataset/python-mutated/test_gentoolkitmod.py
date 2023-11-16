import pytest
from tests.support.case import ModuleCase

@pytest.mark.windows_whitelisted
class GentoolkitModuleTest(ModuleCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up test environment\n        '
        super().setUp()
        ret_grain = self.run_function('grains.item', ['os'])
        if ret_grain['os'] not in 'Gentoo':
            self.skipTest('For Gentoo only')

    def test_revdep_rebuild_true(self):
        if False:
            print('Hello World!')
        ret = self.run_function('gentoolkit.revdep_rebuild')
        self.assertTrue(ret)