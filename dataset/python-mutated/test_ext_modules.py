"""
    :codeauthor: Pedro Algarvio (pedro@algarvio.me)


    integration.loader.ext_modules
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Test Salt's loader regarding external overrides
"""
import os
import time
import pytest
from tests.support.case import ModuleCase
from tests.support.runtests import RUNTIME_VARS

@pytest.mark.windows_whitelisted
class LoaderOverridesTest(ModuleCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.run_function('saltutil.sync_modules')

    @pytest.mark.slow_test
    def test_overridden_internal(self):
        if False:
            return 10
        module = os.path.join(RUNTIME_VARS.TMP, 'rootdir', 'cache', 'files', 'base', '_modules', 'override_test.py')
        tries = 0
        while not os.path.exists(module):
            tries += 1
            if tries > 60:
                break
            time.sleep(1)
        funcs = self.run_function('sys.list_functions')
        self.assertIn('test.ping', funcs)
        self.assertNotIn('brain.left_hemisphere', funcs)
        self.assertIn('test.recho', funcs)
        text = 'foo bar baz quo qux'
        self.assertEqual(self.run_function('test.echo', arg=[text])[::-1], self.run_function('test.recho', arg=[text]))