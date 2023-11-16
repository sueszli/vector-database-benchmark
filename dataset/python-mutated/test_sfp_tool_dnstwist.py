import pytest
import unittest
from modules.sfp_tool_dnstwist import sfp_tool_dnstwist
from sflib import SpiderFoot

@pytest.mark.usefixtures
class TestModuleToolDnstwist(unittest.TestCase):

    def test_opts(self):
        if False:
            for i in range(10):
                print('nop')
        module = sfp_tool_dnstwist()
        self.assertEqual(len(module.opts), len(module.optdescs))

    def test_setup(self):
        if False:
            return 10
        sf = SpiderFoot(self.default_options)
        module = sfp_tool_dnstwist()
        module.setup(sf, dict())

    def test_watchedEvents_should_return_list(self):
        if False:
            while True:
                i = 10
        module = sfp_tool_dnstwist()
        self.assertIsInstance(module.watchedEvents(), list)

    def test_producedEvents_should_return_list(self):
        if False:
            print('Hello World!')
        module = sfp_tool_dnstwist()
        self.assertIsInstance(module.producedEvents(), list)