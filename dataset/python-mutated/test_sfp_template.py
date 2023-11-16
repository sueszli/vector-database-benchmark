import pytest
import unittest
from modules.sfp_template import sfp_template
from sflib import SpiderFoot

@pytest.mark.usefixtures
class TestModuleTemplate(unittest.TestCase):

    def test_opts(self):
        if False:
            for i in range(10):
                print('nop')
        module = sfp_template()
        self.assertEqual(len(module.opts), len(module.optdescs))

    def test_setup(self):
        if False:
            for i in range(10):
                print('nop')
        sf = SpiderFoot(self.default_options)
        module = sfp_template()
        module.setup(sf, dict())

    def test_watchedEvents_should_return_list(self):
        if False:
            return 10
        module = sfp_template()
        self.assertIsInstance(module.watchedEvents(), list)

    def test_producedEvents_should_return_list(self):
        if False:
            print('Hello World!')
        module = sfp_template()
        self.assertIsInstance(module.producedEvents(), list)