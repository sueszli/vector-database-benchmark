"""
    integration.loader.ext_grains
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Test Salt's loader regarding external grains
"""
import os
import time
import pytest
import salt.config
import salt.loader
from tests.support.case import ModuleCase
from tests.support.runtests import RUNTIME_VARS

@pytest.mark.windows_whitelisted
class LoaderGrainsTest(ModuleCase):
    """
    Test the loader standard behavior with external grains
    """

    @pytest.mark.slow_test
    def test_grains_overwrite(self):
        if False:
            while True:
                i = 10
        self.run_function('saltutil.sync_grains')
        module = os.path.join(RUNTIME_VARS.RUNTIME_CONFIGS['minion']['cachedir'], 'files', 'base', '_grains', 'custom_grain2.py')
        tries = 0
        while not os.path.exists(module):
            tries += 1
            if tries > 60:
                self.fail('Failed to found custom grains module in cache path {}'.format(module))
                break
            time.sleep(1)
        grains = self.run_function('grains.items')
        self.assertEqual({'k2': 'v2'}, grains['a_custom'])

@pytest.mark.skip(reason='needs a way to reload minion after config change')
@pytest.mark.windows_whitelisted
class LoaderGrainsMergeTest(ModuleCase):
    """
    Test the loader deep merge behavior with external grains
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.opts = salt.config.minion_config(None)
        self.opts['grains_deep_merge'] = True
        self.assertTrue(self.opts['grains_deep_merge'])
        self.opts['disable_modules'] = ['pillar']
        __grains__ = salt.loader.grains(self.opts)

    def test_grains_merge(self):
        if False:
            i = 10
            return i + 15
        __grain__ = self.run_function('grains.item', ['a_custom'])
        self.assertIn('a_custom', __grain__)
        self.assertEqual({'k1': 'v1', 'k2': 'v2'}, __grain__['a_custom'])