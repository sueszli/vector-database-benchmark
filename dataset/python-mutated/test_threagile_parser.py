from dojo.models import Test
from dojo.tools.threagile.parser import ThreagileParser
from unittests.dojo_test_case import DojoTestCase

class TestThreAgileParser(DojoTestCase):

    def test_non_threagile_file_raises_error(self):
        if False:
            i = 10
            return i + 15
        with open('unittests/scans/threagile/bad_formatted_risks_file.json') as testfile:
            parser = ThreagileParser()
            with self.assertRaises(ValueError) as exc_context:
                parser.get_findings(testfile, Test())
            exc = exc_context.exception
            self.assertEqual('Invalid ThreAgile risks file', str(exc))

    def test_empty_file_returns_no_findings(self):
        if False:
            while True:
                i = 10
        with open('unittests/scans/threagile/empty_file_no_risks.json') as testfile:
            parser = ThreagileParser()
            findings = parser.get_findings(testfile, Test())
            self.assertEqual(0, len(findings))

    def test_file_with_vulnerabilities_returns_correct_findings(self):
        if False:
            i = 10
            return i + 15
        with open('unittests/scans/threagile/risks.json') as testfile:
            parser = ThreagileParser()
            findings = parser.get_findings(testfile, Test())
            self.assertEqual(6, len(findings))
            finding = findings[0]
            self.assertEqual('unguarded-direct-datastore-access', finding.title)
            self.assertEqual('<b>Unguarded Direct Datastore Access</b> of <b>PoliciesRegoStorage</b> by <b>Energon</b> via <b>EnergonToPolicyRegoFileStorage</b>', finding.description)
            self.assertEqual('High', finding.severity)
            self.assertEqual('unguarded-direct-datastore-access@energon-ta>energontopolicyregofilestorage@energon-ta@policies-rego-storage-ta', finding.unique_id_from_tool)
            self.assertEqual(501, finding.cwe)
            self.assertEqual('medium', finding.impact)
            self.assertEqual('policies-rego-storage-ta', finding.component_name)

    def test_in_discussion_is_under_review(self):
        if False:
            print('Hello World!')
        with open('unittests/scans/threagile/risks.json') as testfile:
            parser = ThreagileParser()
            findings = parser.get_findings(testfile, Test())
            finding = findings[1]
            self.assertTrue(finding.under_review)

    def test_accepted_finding_is_accepted(self):
        if False:
            for i in range(10):
                print('nop')
        with open('unittests/scans/threagile/risks.json') as testfile:
            parser = ThreagileParser()
            findings = parser.get_findings(testfile, Test())
            finding = findings[2]
            self.assertTrue(finding.risk_accepted)

    def test_in_progress_is_verified(self):
        if False:
            i = 10
            return i + 15
        with open('unittests/scans/threagile/risks.json') as testfile:
            parser = ThreagileParser()
            findings = parser.get_findings(testfile, Test())
            finding = findings[3]
            self.assertTrue(finding.verified)

    def test_mitigated_is_mitigated(self):
        if False:
            i = 10
            return i + 15
        with open('unittests/scans/threagile/risks.json') as testfile:
            parser = ThreagileParser()
            findings = parser.get_findings(testfile, Test())
            finding = findings[4]
            self.assertTrue(finding.is_mitigated)
            self.assertEqual('some-runtime', finding.component_name)

    def test_false_positive_is_false_positive(self):
        if False:
            for i in range(10):
                print('nop')
        with open('unittests/scans/threagile/risks.json') as testfile:
            parser = ThreagileParser()
            findings = parser.get_findings(testfile, Test())
            finding = findings[5]
            self.assertTrue(finding.false_p)
            self.assertEqual('some-component>some-traffic', finding.component_name)