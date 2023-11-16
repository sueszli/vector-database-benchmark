import os.path
import datetime
from ..dojo_test_case import DojoTestCase, get_unit_tests_path
from dojo.tools.trufflehog3.parser import TruffleHog3Parser
from dojo.models import Test

def sample_path(file_name):
    if False:
        for i in range(10):
            print('nop')
    return os.path.join(get_unit_tests_path() + '/scans/trufflehog3', file_name)

class TestTruffleHog3Parser(DojoTestCase):

    def test_zero_vulns(self):
        if False:
            return 10
        test_file = open(sample_path('zero_vulns.json'))
        parser = TruffleHog3Parser()
        findings = parser.get_findings(test_file, Test())
        self.assertEqual(len(findings), 0)

    def test_many_vulns_legacy(self):
        if False:
            i = 10
            return i + 15
        test_file = open(sample_path('many_vulns_legacy.json'))
        parser = TruffleHog3Parser()
        findings = parser.get_findings(test_file, Test())
        self.assertEqual(len(findings), 7)
        finding = findings[0]
        self.assertEqual('High', finding.severity)
        self.assertEqual(798, finding.cwe)
        self.assertEqual('fixtures/users.json', finding.file_path)
        self.assertEqual(datetime.date, type(finding.date))
        self.assertEqual(7, finding.nb_occurences)

    def test_many_vulns2_legacy(self):
        if False:
            i = 10
            return i + 15
        test_file = open(sample_path('many_vulns2_legacy.json'))
        parser = TruffleHog3Parser()
        findings = parser.get_findings(test_file, Test())
        self.assertEqual(len(findings), 27)
        finding = findings[0]
        self.assertEqual('High', finding.severity)
        self.assertEqual(798, finding.cwe)
        self.assertEqual('test_all.py', finding.file_path)
        self.assertEqual(8, finding.nb_occurences)

    def test_many_vulns_current(self):
        if False:
            return 10
        test_file = open(sample_path('many_vulns_current.json'))
        parser = TruffleHog3Parser()
        findings = parser.get_findings(test_file, Test())
        self.assertEqual(len(findings), 3)
        finding = findings[0]
        self.assertEqual('High Entropy found in docker/Dockerfile', finding.title)
        self.assertEqual(798, finding.cwe)
        description = '**Secret:** 964a1afa20dd4a3723002560124dd96f2a9e853f7ef5b86f5c2354af336fca37\n**Context:**\n    3: +FROM python:3.9.7-alpine@sha256:964a1afa20dd4a3723002560124dd96f2a9e853f7ef5b86f5c2354af336fca37\n**Branch:** python-ab08dd9\n**Commit message:** Bump python from 3.9.7-alpine to 3.10.0-alpine\n**Commit hash:** 9c3f4d641d14eba2740febccd902cde300218a8d\n**Commit date:** 2021-10-08T20:14:27+02:00'
        self.assertEqual(description, finding.description)
        self.assertEqual('High', finding.severity)
        self.assertEqual('docker/Dockerfile', finding.file_path)
        self.assertEqual(3, finding.line)
        self.assertEqual(1, finding.nb_occurences)
        finding = findings[1]
        self.assertEqual('High Entropy found in docker/Dockerfile', finding.title)
        self.assertEqual(798, finding.cwe)
        self.maxDiff = None
        self.assertIn('\n\n***\n\n', finding.description)
        self.assertEqual('Medium', finding.severity)
        self.assertEqual('docker/Dockerfile', finding.file_path)
        self.assertEqual(2, finding.line)
        self.assertEqual(2, finding.nb_occurences)
        finding = findings[2]
        self.assertEqual('High Entropy found in env-file.txt', finding.title)
        self.assertEqual(798, finding.cwe)
        description = '**Secret:** 44c45225cf94e58d0c86f0a31051eb7c52c8f78f\n**Context:**\n    10: DD_API_KEY=44c45225cf94e58d0c86f0a31051eb7c52c8f78f\n    11: second line of context'
        self.assertEqual(description, finding.description)
        self.assertEqual('Low', finding.severity)
        self.assertEqual('env-file.txt', finding.file_path)
        self.assertEqual(10, finding.line)
        self.assertEqual(1, finding.nb_occurences)