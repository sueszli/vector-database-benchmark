from ..dojo_test_case import DojoTestCase
from dojo.tools.wfuzz.parser import WFuzzParser
from dojo.models import Test

class TestWFuzzParser(DojoTestCase):

    def test_parse_no_findings(self):
        if False:
            for i in range(10):
                print('nop')
        testfile = open('unittests/scans/wfuzz/no_findings.json')
        parser = WFuzzParser()
        findings = parser.get_findings(testfile, Test())
        self.assertEqual(0, len(findings))

    def test_parse_one_finding(self):
        if False:
            while True:
                i = 10
        testfile = open('unittests/scans/wfuzz/one_finding.json')
        parser = WFuzzParser()
        findings = parser.get_findings(testfile, Test())
        for finding in findings:
            for endpoint in finding.unsaved_endpoints:
                endpoint.clean()
        self.assertEqual(1, len(findings))

    def test_parse_many_finding(self):
        if False:
            print('Hello World!')
        testfile = open('unittests/scans/wfuzz/many_findings.json')
        parser = WFuzzParser()
        findings = parser.get_findings(testfile, Test())
        for finding in findings:
            for endpoint in finding.unsaved_endpoints:
                endpoint.clean()
        self.assertEqual(4, len(findings))

    def test_one_dup_finding(self):
        if False:
            return 10
        testfile = open('unittests/scans/wfuzz/one_dup_finding.json')
        parser = WFuzzParser()
        findings = parser.get_findings(testfile, Test())
        for finding in findings:
            for endpoint in finding.unsaved_endpoints:
                endpoint.clean()
        self.assertEqual(4, len(findings))