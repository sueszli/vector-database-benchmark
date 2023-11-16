from os import path
from ..dojo_test_case import DojoTestCase
from dojo.models import Test
from dojo.tools.neuvector_compliance.parser import NeuVectorComplianceParser

class TestNeuVectorComplianceParser(DojoTestCase):

    def test_parse_file_with_no_vuln(self):
        if False:
            i = 10
            return i + 15
        testfile = open(path.join(path.dirname(__file__), '../scans/neuvector_compliance/no_vuln.json'))
        parser = NeuVectorComplianceParser()
        findings = parser.get_findings(testfile, Test())
        testfile.close()
        self.assertEqual(0, len(findings))

    def test_parse_file_with_one_vuln(self):
        if False:
            return 10
        testfile = open(path.join(path.dirname(__file__), '../scans/neuvector_compliance/one_vuln.json'))
        parser = NeuVectorComplianceParser()
        findings = parser.get_findings(testfile, Test())
        testfile.close()
        self.assertEqual(1, len(findings))
        self.assertEqual('docker_D.1.1.11', findings[0].vuln_id_from_tool)

    def test_parse_file_with_many_vulns(self):
        if False:
            while True:
                i = 10
        testfile = open(path.join(path.dirname(__file__), '../scans/neuvector_compliance/many_vulns.json'))
        parser = NeuVectorComplianceParser()
        findings = parser.get_findings(testfile, Test())
        testfile.close()
        self.assertEqual(6, len(findings))