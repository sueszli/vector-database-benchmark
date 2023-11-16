from ..dojo_test_case import DojoTestCase
from dojo.tools.brakeman.parser import BrakemanParser
from dojo.models import Test

class TestBrakemanParser(DojoTestCase):

    def test_parse_file_no_finding(self):
        if False:
            for i in range(10):
                print('nop')
        testfile = open('unittests/scans/brakeman/no_finding.json')
        parser = BrakemanParser()
        findings = parser.get_findings(testfile, Test())
        self.assertEqual(0, len(findings))

    def test_parse_file_has_two_findings(self):
        if False:
            print('Hello World!')
        testfile = open('unittests/scans/brakeman/two_findings.json')
        parser = BrakemanParser()
        findings = parser.get_findings(testfile, Test())
        testfile.close()
        self.assertEqual(2, len(findings))

    def test_parse_file_has_many_findings(self):
        if False:
            for i in range(10):
                print('nop')
        testfile = open('unittests/scans/brakeman/many_findings.json')
        parser = BrakemanParser()
        findings = parser.get_findings(testfile, Test())
        testfile.close()
        self.assertEqual(18, len(findings))