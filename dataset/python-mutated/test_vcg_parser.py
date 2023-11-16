import io
import csv
from defusedxml import ElementTree
from ..dojo_test_case import DojoTestCase
from dojo.models import Test
from dojo.tools.vcg.parser import VCGCsvParser
from dojo.tools.vcg.parser import VCGParser
from dojo.tools.vcg.parser import VCGXmlParser

class TestFile(object):

    def read(self):
        if False:
            i = 10
            return i + 15
        return self.content

    def __init__(self, name, content):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.content = content

class TestVCGXmlParser(DojoTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.parser = VCGXmlParser()

    def test_parse_no_content_no_findings(self):
        if False:
            i = 10
            return i + 15
        results = self.parser.parse(None, Test())
        self.assertEqual(0, len(results))

    def test_parse_single_finding(self):
        if False:
            print('Hello World!')
        single_finding = '<?xml version="1.0" encoding="utf-8"?>\n        <!--XML Export of VCG Results for directory: C:\\Projects\\WebGoat.Net. Scanned for C# security issues.-->\n        <CodeIssueCollection>\n        <CodeIssue>\n        <Priority>6</Priority>\n        <Severity>Suspicious Comment</Severity>\n        <Title>Comment Indicates Potentially Unfinished Code</Title>\n        <Description>The comment includes some wording which indicates that the developer regards\n        it as unfinished or does not trust it to work correctly.</Description>\n        <FileName>Findings.xml</FileName>\n        <Line>21</Line>\n        <CodeLine>TODO: Check the Code</CodeLine>\n        <Checked>False</Checked>\n        <CheckColour>LawnGreen</CheckColour>\n        </CodeIssue>\n        </CodeIssueCollection>'
        results = self.parser.parse(single_finding, Test())
        self.assertEqual(1, len(results))

    def test_parse_multiple_findings(self):
        if False:
            i = 10
            return i + 15
        findings = '<?xml version="1.0" encoding="utf-8"?>\n        <!--XML Export of VCG Results for directory: C:\\Projects\\WebGoat.Net. Scanned for C# security issues.-->\n        <CodeIssueCollection>\n        <CodeIssue>\n        <Priority>6</Priority>\n        <Severity>Suspicious Comment</Severity>\n        <Title>Comment Indicates Potentially Unfinished Code</Title>\n        <Description>The comment includes some wording which indicates that the developer regards\n        it as unfinished or does not trust it to work correctly.</Description>\n        <FileName>Findings.xml</FileName>\n        <Line>21</Line>\n        <CodeLine>TODO: Check the Code</CodeLine>\n        <Checked>False</Checked>\n        <CheckColour>LawnGreen</CheckColour>\n        </CodeIssue>\n        <CodeIssue>\n        <Priority>6</Priority>\n        <Severity>Suspicious Comment</Severity>\n        <Title>Comment Indicates Potentially Unfinished Code</Title>\n        <Description>The comment includes some wording which indicates that the developer regards\n        it as unfinished or does not trust it to work correctly.</Description>\n        <FileName>Findings.xml</FileName>\n        <Line>62</Line>\n        <CodeLine>TODO: Check the Code</CodeLine>\n        <Checked>False</Checked>\n        <CheckColour>LawnGreen</CheckColour>\n        </CodeIssue>\n        </CodeIssueCollection>'
        results = self.parser.parse(findings, Test())
        self.assertEqual(2, len(results))

    def test_parse_duplicate_findings_dedupes(self):
        if False:
            return 10
        duplicate_finding = '<?xml version="1.0" encoding="utf-8"?>\n        <!--XML Export of VCG Results for directory: C:\\Projects\\WebGoat.Net. Scanned for C# security issues.-->\n        <CodeIssueCollection>\n        <CodeIssue>\n        <Priority>6</Priority>\n        <Severity>Suspicious Comment</Severity>\n        <Title>Comment Indicates Potentially Unfinished Code</Title>\n        <Description>The comment includes some wording which indicates that the developer regards\n        it as unfinished or does not trust it to work correctly.</Description>\n        <FileName>Findings.xml</FileName>\n        <Line>21</Line>\n        <CodeLine>TODO: Check the Code</CodeLine>\n        <Checked>False</Checked>\n        <CheckColour>LawnGreen</CheckColour>\n        </CodeIssue>\n        <CodeIssue>\n        <Priority>6</Priority>\n        <Severity>Suspicious Comment</Severity>\n        <Title>Comment Indicates Potentially Unfinished Code</Title>\n        <Description>The comment includes some wording which indicates that the developer regards\n        it as unfinished or does not trust it to work correctly.</Description>\n        <FileName>Findings.xml</FileName>\n        <Line>21</Line>\n        <CodeLine>TODO: Check the Code</CodeLine>\n        <Checked>False</Checked>\n        <CheckColour>LawnGreen</CheckColour>\n        </CodeIssue>\n        </CodeIssueCollection>'
        results = self.parser.parse(duplicate_finding, Test())
        self.assertEqual(1, len(results))

    def test_parseissuexml_with_no_issue_has_no_finding(self):
        if False:
            print('Hello World!')
        self.assertIsNone(self.parser.parse_issue(None, Test()))

    def test_parseissuexml_with_issue_has_finding(self):
        if False:
            print('Hello World!')
        single_finding = '<?xml version="1.0" encoding="utf-8"?>\n        <!--XML Export of VCG Results for directory: C:\\Projects\\WebGoat.Net. Scanned for C# security issues.-->\n        <CodeIssueCollection>\n        <CodeIssue>\n        <Priority>6</Priority>\n        <Severity>Suspicious Comment</Severity>\n        <Title>Comment Indicates Potentially Unfinished Code</Title>\n        <Description>The comment includes some wording which indicates that the developer regards\n        it as unfinished or does not trust it to work correctly.</Description>\n        <FileName>Findings.xml</FileName>\n        <Line>21</Line>\n        <CodeLine>TODO: Check the Code</CodeLine>\n        <Checked>False</Checked>\n        <CheckColour>LawnGreen</CheckColour>\n        </CodeIssue>\n        </CodeIssueCollection>'
        vcgscan = ElementTree.fromstring(single_finding)
        finding = self.parser.parse_issue(vcgscan.findall('CodeIssue')[0], Test())
        self.assertEqual('Info', finding.severity)
        self.assertEqual('Comment Indicates Potentially Unfinished Code', finding.title)

class TestVCGCsvParser(DojoTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.parser = VCGCsvParser()

    def test_parse_no_csv_content_no_findings(self):
        if False:
            print('Hello World!')
        findings = ''
        results = self.parser.parse(findings, Test())
        self.assertEqual(0, len(results))

    def test_parse_single_finding_single_result(self):
        if False:
            while True:
                i = 10
        findings = '6,Suspicious Comment,"Comment Indicates Potentially Unfinished Code","The comment includes some wording which indicates that the developer regards it as unfinished or does not trust it to work correctly.",C:\\Projects\\WebGoat.Net\\Core\\Cart.cs,16,"TODO: Refactor this. Use LINQ with aggregation to get SUM.",False,"LawnGreen'
        results = self.parser.parse(findings, Test())
        self.assertEqual(1, len(results))

    def test_parse_multiple_findings_multiple_results(self):
        if False:
            for i in range(10):
                print('nop')
        findings = '6,Suspicious Comment,"Comment Indicates Potentially Unfinished Code","The comment includes some wording which indicates that the developer regards it as unfinished or does not trust it to work correctly.",C:\\Projects\\WebGoat.Net\\Core\\Cart.cs,16,"TODO: Refactor this. Use LINQ with aggregation to get SUM.",False,"LawnGreen"\n6,Suspicious Comment,"Comment Indicates Potentially Unfinished Code","The comment includes some wording which indicates that the developer regards it as unfinished or does not trust it to work correctly.",C:\\Projects\\WebGoat.Net\\Core\\Cart.cs,41,"TODO: Add ability to delete an orderDetail and to change quantities.",False,"LawnGreen'
        results = self.parser.parse(findings, Test())
        self.assertEqual(2, len(results))

    def test_parse_duplicate_findings_deduped_results(self):
        if False:
            while True:
                i = 10
        findings = '6,Suspicious Comment,"Comment Indicates Potentially Unfinished Code","The comment includes some wording which indicates that the developer regards it as unfinished or does not trust it to work correctly.",C:\\Projects\\WebGoat.Net\\Core\\Cart.cs,16,"TODO: Refactor this. Use LINQ with aggregation to get SUM.",False,"LawnGreen"\n6,Suspicious Comment,"Comment Indicates Potentially Unfinished Code","The comment includes some wording which indicates that the developer regards it as unfinished or does not trust it to work correctly.",C:\\Projects\\WebGoat.Net\\Core\\Cart.cs,16,"TODO: Refactor this. Use LINQ with aggregation to get SUM.",False,"LawnGreen'
        results = self.parser.parse(findings, Test())
        self.assertEqual(1, len(results))

    def test_parseissuerow_with_no_row_has_no_finding(self):
        if False:
            return 10
        finding = self.parser.parse_issue(None, Test())
        self.assertIsNone(finding)

    def test_parseissuerow_with_empty_row_has_no_finding(self):
        if False:
            for i in range(10):
                print('nop')
        row = dict()
        finding = self.parser.parse_issue(row, Test())
        self.assertIsNone(finding)

    def test_parseissuerow_with_row_has_finding(self):
        if False:
            return 10
        findings = '6,Suspicious Comment,"Comment Indicates Potentially Unfinished Code","The comment includes some wording which indicates that the developer regards it as unfinished or does not trust it to work correctly.",C:\\Projects\\WebGoat.Net\\Core\\Cart.cs,16,"TODO: Refactor this. Use LINQ with aggregation to get SUM.",False,"LawnGreen'
        reader = csv.reader(io.StringIO(findings), delimiter=',', quotechar='"')
        finding = None
        for row in reader:
            finding = self.parser.parse_issue(row, Test())
        self.assertIsNotNone(finding)
        self.assertEqual('Info', finding.severity)
        self.assertEqual('Comment Indicates Potentially Unfinished Code', finding.title)

class TestVCGImport(DojoTestCase):

    def test_can_parse_xml(self):
        if False:
            for i in range(10):
                print('nop')
        content = '<?xml version="1.0" encoding="utf-8"?>\n        <!--XML Export of VCG Results for directory: C:\\Projects\\WebGoat.Net. Scanned for C# security issues.-->\n        <CodeIssueCollection>\n        <CodeIssue>\n        <Priority>6</Priority>\n        <Severity>Suspicious Comment</Severity>\n        <Title>Comment Indicates Potentially Unfinished Code</Title>\n        <Description>The comment includes some wording which indicates that the developer regards\n        it as unfinished or does not trust it to work correctly.</Description>\n        <FileName>Findings.xml</FileName>\n        <Line>21</Line>\n        <CodeLine>TODO: Check the Code</CodeLine>\n        <Checked>False</Checked>\n        <CheckColour>LawnGreen</CheckColour>\n        </CodeIssue>\n        </CodeIssueCollection>'
        filename = TestFile('data.xml', content)
        parser = VCGParser()
        findings = parser.get_findings(filename, Test())
        self.assertEqual(1, len(findings))

    def test_can_parse_csv(self):
        if False:
            return 10
        content = '6,Suspicious Comment,"Comment Indicates Potentially Unfinished Code","The comment includes some wording which indicates that the developer regards it as unfinished or does not trust it to work correctly.",C:\\Projects\\WebGoat.Net\\Core\\Cart.cs,16,"TODO: Refactor this. Use LINQ with aggregation to get SUM.",False,"LawnGreen'
        filename = TestFile('data.csv', content)
        parser = VCGParser()
        findings = parser.get_findings(filename, Test())
        self.assertEqual(1, len(findings))