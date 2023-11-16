from ..dojo_test_case import DojoTestCase
from dojo.tools.eslint.parser import ESLintParser
from dojo.models import Test

class TestESLintParser(DojoTestCase):

    def test_parse_file_has_two_findings(self):
        if False:
            while True:
                i = 10
        testfile = open('unittests/scans/eslint/scan.json')
        parser = ESLintParser()
        findings = parser.get_findings(testfile, Test())
        testfile.close()
        self.assertEqual(2, len(findings))

    def test_parse_empty_file(self):
        if False:
            for i in range(10):
                print('nop')
        testfile = open('unittests/scans/eslint/empty.json')
        parser = ESLintParser()
        findings = parser.get_findings(testfile, Test())
        testfile.close()
        self.assertEqual(0, len(findings))

    def test_parse_file_with_no_finding(self):
        if False:
            print('Hello World!')
        testfile = open('unittests/scans/eslint/no_finding.json')
        parser = ESLintParser()
        findings = parser.get_findings(testfile, Test())
        testfile.close()
        self.assertEqual(0, len(findings))