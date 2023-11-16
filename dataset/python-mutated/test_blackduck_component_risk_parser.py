from ..dojo_test_case import DojoTestCase, get_unit_tests_path
from dojo.tools.blackduck_component_risk.parser import BlackduckComponentRiskParser
from dojo.models import Test
from pathlib import Path

class TestBlackduckComponentRiskParser(DojoTestCase):

    def test_blackduck_enhanced_zip_upload(self):
        if False:
            return 10
        testfile = Path(get_unit_tests_path() + '/scans/blackduck_component_risk/blackduck_hub_component_risk.zip')
        parser = BlackduckComponentRiskParser()
        findings = parser.get_findings(testfile, Test())
        self.assertEqual(12, len(findings))