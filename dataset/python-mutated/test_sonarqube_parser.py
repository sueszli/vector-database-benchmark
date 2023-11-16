from ..dojo_test_case import DojoTestCase, get_unit_tests_path
from dojo.models import Test, Engagement, Product
from dojo.tools.sonarqube.parser import SonarQubeParser

class TestSonarQubeParser(DojoTestCase):

    def init(self, reportFilename):
        if False:
            print('Hello World!')
        my_file_handle = open(reportFilename)
        product = Product()
        engagement = Engagement()
        test = Test()
        engagement.product = product
        test.engagement = engagement
        return (my_file_handle, product, engagement, test)

    def test_file_name_aggregated_parse_file_with_no_vulnerabilities_has_no_findings(self):
        if False:
            while True:
                i = 10
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar-no-finding.html')
        parser = SonarQubeParser()
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(0, len(findings))

    def test_detailed_parse_file_with_no_vulnerabilities_has_no_findings(self):
        if False:
            print('Hello World!')
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar-no-finding.html')
        parser = SonarQubeParser()
        parser.set_mode('detailed')
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(0, len(findings))

    def test_file_name_aggregated_parse_file_with_single_vulnerability_has_single_finding(self):
        if False:
            i = 10
            return i + 15
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar-single-finding.html')
        parser = SonarQubeParser()
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(1, len(findings))
        item = findings[0]
        self.assertEqual(str, type(item.description))
        self.assertMultiLineEqual('Because it is easy to extract strings from a compiled application, credentials should never be hard-coded. Do so, and they\'re almost guaranteed to\nend up in the hands of an attacker. This is particularly true for applications that are distributed.\nCredentials should be stored outside of the code in a strongly-protected encrypted configuration file or database.\nIt\'s recommended to customize the configuration of this rule with additional credential words such as "oauthToken", "secret", ...\n**Noncompliant Code Example**\n\nConnection conn = null;\ntry {\n  conn = DriverManager.getConnection("jdbc:mysql://localhost/test?" +\n        "user=steve&amp;password=blue"); // Noncompliant\n  String uname = "steve";\n  String password = "blue";\n  conn = DriverManager.getConnection("jdbc:mysql://localhost/test?" +\n        "user=" + uname + "&amp;password=" + password); // Noncompliant\n\n  java.net.PasswordAuthentication pa = new java.net.PasswordAuthentication("userName", "1234".toCharArray());  // Noncompliant\n\n**Compliant Solution**\n\nConnection conn = null;\ntry {\n  String uname = getEncryptedUser();\n  String password = getEncryptedPass();\n  conn = DriverManager.getConnection("jdbc:mysql://localhost/test?" +\n        "user=" + uname + "&amp;password=" + password);\n\n-----\nOccurences:\nLine: 66', item.description)
        self.assertIsNone(item.line)
        self.assertIsNone(item.unique_id_from_tool)
        self.assertEqual(int, type(item.nb_occurences))
        self.assertEqual(1, item.nb_occurences)

    def test_detailed_parse_file_with_single_vulnerability_has_single_finding(self):
        if False:
            print('Hello World!')
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar-single-finding.html')
        parser = SonarQubeParser()
        parser.set_mode('detailed')
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(1, len(findings))
        item = findings[0]
        self.assertEqual(str, type(item.description))
        self.assertMultiLineEqual('Because it is easy to extract strings from a compiled application, credentials should never be hard-coded. Do so, and they\'re almost guaranteed to\nend up in the hands of an attacker. This is particularly true for applications that are distributed.\nCredentials should be stored outside of the code in a strongly-protected encrypted configuration file or database.\nIt\'s recommended to customize the configuration of this rule with additional credential words such as "oauthToken", "secret", ...\n**Noncompliant Code Example**\n\nConnection conn = null;\ntry {\n  conn = DriverManager.getConnection("jdbc:mysql://localhost/test?" +\n        "user=steve&amp;password=blue"); // Noncompliant\n  String uname = "steve";\n  String password = "blue";\n  conn = DriverManager.getConnection("jdbc:mysql://localhost/test?" +\n        "user=" + uname + "&amp;password=" + password); // Noncompliant\n\n  java.net.PasswordAuthentication pa = new java.net.PasswordAuthentication("userName", "1234".toCharArray());  // Noncompliant\n\n**Compliant Solution**\n\nConnection conn = null;\ntry {\n  String uname = getEncryptedUser();\n  String password = getEncryptedPass();\n  conn = DriverManager.getConnection("jdbc:mysql://localhost/test?" +\n        "user=" + uname + "&amp;password=" + password);', item.description)
        self.assertEqual(str, type(item.line))
        self.assertEqual('66', item.line)
        self.assertEqual(str, type(item.unique_id_from_tool))
        self.assertEqual('AWK40IMu-pl6AHs22MnV', item.unique_id_from_tool)

    def check_parse_file_with_single_vulnerability_has_single_finding(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(1, len(findings))
        item = findings[0]
        self.assertEqual(str, type(findings[0].title))
        self.assertEqual('Credentials should not be hard-coded', item.title)
        self.assertEqual(int, type(item.cwe))
        self.assertEqual(798, item.cwe)
        self.assertEqual(bool, type(item.active))
        self.assertEqual(False, item.active)
        self.assertEqual(bool, type(item.verified))
        self.assertEqual(False, item.verified)
        self.assertEqual(str, type(item.severity))
        self.assertEqual('Critical', item.severity)
        self.assertEqual(str, type(item.mitigation))
        self.assertEqual("'PASSWORD' detected in this expression, review this potentially hardcoded credential.", item.mitigation)
        self.assertEqual(str, type(item.references))
        self.assertMultiLineEqual('squid:S2068\nOWASP Top 10 2017 Category A2\nMITRE, CWE-798\nMITRE, CWE-259\nCERT, MSC03-J.\nSANS Top 25\nHard Coded Password', item.references)
        self.assertEqual(str, type(item.file_path))
        self.assertEqual('modules/jdbc-pool/src/main/java/org/apache/tomcat/jdbc/pool/DataSourceFactory.java', item.file_path)
        self.assertEqual(bool, type(item.static_finding))
        self.assertEqual(True, item.static_finding)
        self.assertEqual(bool, type(item.dynamic_finding))
        self.assertEqual(False, item.dynamic_finding)

    def test_detailed_parse_file_with_multiple_vulnerabilities_has_multiple_findings(self):
        if False:
            for i in range(10):
                print('nop')
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar-6-findings.html')
        parser = SonarQubeParser()
        parser.set_mode('detailed')
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(6, len(findings))

    def test_file_name_aggregated_parse_file_with_multiple_vulnerabilities_has_multiple_findings(self):
        if False:
            while True:
                i = 10
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar-6-findings.html')
        parser = SonarQubeParser()
        parser.set_mode('detailed')
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(6, len(findings))

    def test_detailed_parse_file_with_table_in_table(self):
        if False:
            return 10
        'Test parsing when the vulnerability details include a table, with tr and td that should be ignored when looking for list of rules'
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar-table-in-table.html')
        parser = SonarQubeParser()
        parser.set_mode('detailed')
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(1, len(findings))
        item = findings[0]
        self.assertEqual(str, type(findings[0].title))
        self.assertEqual('"clone" should not be overridden', item.title)
        self.assertEqual(int, type(item.cwe))
        self.assertEqual(0, item.cwe)
        self.assertEqual(bool, type(item.active))
        self.assertEqual(True, item.active)
        self.assertEqual(bool, type(item.verified))
        self.assertEqual(False, item.verified)
        self.assertEqual(str, type(item.description))
        self.assertMultiLineEqual('Many consider clone and Cloneable broken in Java, largely because the rules for overriding clone are tricky\nand difficult to get right, according to Joshua Bloch:\n\n  Object\'s clone method is very tricky. It\'s based on field copies, and it\'s "extra-linguistic." It creates an object without calling a constructor.\n  There are no guarantees that it preserves the invariants established by the constructors. There have been lots of bugs over the years, both in and\n  outside Sun, stemming from the fact that if you just call super.clone repeatedly up the chain until you have cloned an object, you have a shallow\n  copy of the object. The clone generally shares state with the object being cloned. If that state is mutable, you don\'t have two independent objects.\n  If you modify one, the other changes as well. And all of a sudden, you get random behavior.\n\nA copy constructor or copy factory should be used instead.\nThis rule raises an issue when clone is overridden, whether or not Cloneable is implemented.\n**Noncompliant Code Example**\n\npublic class MyClass {\n  // ...\n\n  public Object clone() { // Noncompliant\n    //...\n  }\n}\n\n**Compliant Solution**\n\npublic class MyClass {\n  // ...\n\n  MyClass (MyClass source) {\n    //...\n  }\n}', item.description)
        self.assertEqual(str, type(item.severity))
        self.assertEqual('Critical', item.severity)
        self.assertEqual(str, type(item.mitigation))
        self.assertEqual('Remove this "clone" implementation; use a copy constructor or copy factory instead.', item.mitigation)
        self.assertEqual(str, type(item.references))
        self.assertMultiLineEqual('squid:S2975\nCopy Constructor versus Cloning\nS2157\nS1182', item.references)
        self.assertEqual(str, type(item.file_path))
        self.assertEqual('java/org/apache/catalina/util/URLEncoder.java', item.file_path)
        self.assertEqual(str, type(item.line))
        self.assertEqual('190', item.line)
        self.assertEqual(str, type(item.unique_id_from_tool))
        self.assertEqual('AWK40IMu-pl6AHs22MnV', item.unique_id_from_tool)
        self.assertEqual(bool, type(item.static_finding))
        self.assertEqual(True, item.static_finding)
        self.assertEqual(bool, type(item.dynamic_finding))
        self.assertEqual(False, item.dynamic_finding)

    def test_detailed_parse_file_with_rule_undefined(self):
        if False:
            print('Hello World!')
        "the vulnerability's rule is not in the list of rules"
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar-rule-undefined.html')
        parser = SonarQubeParser()
        parser.set_mode('detailed')
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(1, len(findings))
        item = findings[0]
        self.assertEqual(str, type(findings[0].title))
        self.assertEqual('"clone" should not be overridden', item.title)
        self.assertEqual(int, type(item.cwe))
        self.assertEqual(0, item.cwe)
        self.assertEqual(bool, type(item.active))
        self.assertEqual(True, item.active)
        self.assertEqual(bool, type(item.verified))
        self.assertEqual(False, item.verified)
        self.assertEqual(str, type(item.description))
        self.assertEqual('No description provided', item.description)
        self.assertEqual(str, type(item.severity))
        self.assertEqual('Critical', item.severity)
        self.assertEqual(str, type(item.mitigation))
        self.assertEqual('Remove this "clone" implementation; use a copy constructor or copy factory instead.', item.mitigation)
        self.assertEqual(str, type(item.references))
        self.assertEqual('', item.references)
        self.assertEqual(str, type(item.file_path))
        self.assertEqual('java/org/apache/catalina/util/URLEncoder.java', item.file_path)
        self.assertEqual(str, type(item.line))
        self.assertEqual('190', item.line)
        self.assertEqual(str, type(item.unique_id_from_tool))
        self.assertEqual('AWK40IMu-pl6AHs22MnV', item.unique_id_from_tool)
        self.assertEqual(bool, type(item.static_finding))
        self.assertEqual(True, item.static_finding)
        self.assertEqual(bool, type(item.dynamic_finding))
        self.assertEqual(False, item.dynamic_finding)

    def test_file_name_aggregated_parse_file_with_vuln_on_same_filename(self):
        if False:
            return 10
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar-4-findings-3-to-aggregate.html')
        parser = SonarQubeParser()
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(2, len(findings))
        item1 = findings[0]
        item2 = findings[1]
        if item1.nb_occurences == 3:
            aggregatedItem = item1
            self.assertEqual(int, type(item2.nb_occurences))
            self.assertEqual(1, item2.nb_occurences)
        elif item2.nb_occurences == 3:
            aggregatedItem = item2
            self.assertEqual(int, type(item1.nb_occurences))
            self.assertEqual(1, item1.nb_occurences)
        else:
            self.fail('cannot find aggregated item')
        self.assertEqual(str, type(aggregatedItem.description))
        self.assertMultiLineEqual('Because it is easy to extract strings from a compiled application, credentials should never be hard-coded. Do so, and they\'re almost guaranteed to\nend up in the hands of an attacker. This is particularly true for applications that are distributed.\nCredentials should be stored outside of the code in a strongly-protected encrypted configuration file or database.\n**Noncompliant Code Example**\n\nConnection conn = null;\ntry {\n  conn = DriverManager.getConnection("jdbc:mysql://localhost/test?" +\n        "user=steve&amp;password=blue"); // Noncompliant\n  String uname = "steve";\n  String password = "blue";\n  conn = DriverManager.getConnection("jdbc:mysql://localhost/test?" +\n        "user=" + uname + "&amp;password=" + password); // Noncompliant\n\n  java.net.PasswordAuthentication pa = new java.net.PasswordAuthentication("userName", "1234".toCharArray());  // Noncompliant\n\n**Compliant Solution**\n\nConnection conn = null;\ntry {\n  String uname = getEncryptedUser();\n  String password = getEncryptedPass();\n  conn = DriverManager.getConnection("jdbc:mysql://localhost/test?" +\n        "user=" + uname + "&amp;password=" + password);\n\n-----\nOccurences:\nLine: 12\nLine: 13\nLine: 14', aggregatedItem.description)
        self.assertIsNone(aggregatedItem.line)
        self.assertIsNone(aggregatedItem.unique_id_from_tool)
        self.assertEqual(int, type(aggregatedItem.nb_occurences))

    def test_detailed_parse_file_with_vuln_on_same_filename(self):
        if False:
            while True:
                i = 10
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar-4-findings-3-to-aggregate.html')
        parser = SonarQubeParser()
        parser.set_mode('detailed')
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(4, len(findings))

    def test_detailed_parse_file_with_vuln_issue_3725(self):
        if False:
            while True:
                i = 10
        'SonarQube Scan detailed - report that crash\n        see: https://github.com/DefectDojo/django-DefectDojo/issues/3725\n        '
        (my_file_handle, product, engagement, test) = self.init(get_unit_tests_path() + '/scans/sonarqube/sonar.html')
        parser = SonarQubeParser()
        parser.set_mode('detailed')
        findings = parser.get_findings(my_file_handle, test)
        self.assertEqual(322, len(findings))