import pytest
import unittest

from modules.sfp_zonefiles import sfp_zonefiles
from sflib import SpiderFoot


@pytest.mark.usefixtures
class TestModuleZoneFiles(unittest.TestCase):

    def test_opts(self):
        module = sfp_zonefiles()
        self.assertEqual(len(module.opts), len(module.optdescs))

    def test_setup(self):
        sf = SpiderFoot(self.default_options)
        module = sfp_zonefiles()
        module.setup(sf, dict())

    def test_watchedEvents_should_return_list(self):
        module = sfp_zonefiles()
        self.assertIsInstance(module.watchedEvents(), list)

    def test_producedEvents_should_return_list(self):
        module = sfp_zonefiles()
        self.assertIsInstance(module.producedEvents(), list)

    def test_parseApiResponse_nonfatal_http_response_code_should_not_set_errorState(self):
        sf = SpiderFoot(self.default_options)

        http_codes = ["200"]
        for code in http_codes:
            with self.subTest(code=code):
                module = sfp_zonefiles()
                module.setup(sf, dict())
                result = module.parseApiResponse({"code": code, "content": None})
                self.assertIsNone(result)
                self.assertFalse(module.errorState)

    def test_parseApiResponse_fatal_http_response_error_code_should_set_errorState(self):
        sf = SpiderFoot(self.default_options)

        http_codes = ["401", "403", "404", "429", "500", "502", "503"]
        for code in http_codes:
            with self.subTest(code=code):
                module = sfp_zonefiles()
                module.setup(sf, dict())
                result = module.parseApiResponse({"code": code, "content": None})
                self.assertIsNone(result)
                self.assertTrue(module.errorState)
