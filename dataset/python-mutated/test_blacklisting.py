import testtools
from bandit.core import blacklisting

class BlacklistingTests(testtools.TestCase):

    def test_report_issue(self):
        if False:
            return 10
        data = {'level': 'HIGH', 'message': 'test {name}', 'id': 'B000'}
        issue = blacklisting.report_issue(data, 'name')
        issue_dict = issue.as_dict(with_code=False)
        self.assertIsInstance(issue_dict, dict)
        self.assertEqual('B000', issue_dict['test_id'])
        self.assertEqual('HIGH', issue_dict['issue_severity'])
        self.assertEqual({}, issue_dict['issue_cwe'])
        self.assertEqual('HIGH', issue_dict['issue_confidence'])
        self.assertEqual('test name', issue_dict['issue_text'])

    def test_report_issue_defaults(self):
        if False:
            print('Hello World!')
        data = {'message': 'test {name}'}
        issue = blacklisting.report_issue(data, 'name')
        issue_dict = issue.as_dict(with_code=False)
        self.assertIsInstance(issue_dict, dict)
        self.assertEqual('LEGACY', issue_dict['test_id'])
        self.assertEqual('MEDIUM', issue_dict['issue_severity'])
        self.assertEqual({}, issue_dict['issue_cwe'])
        self.assertEqual('HIGH', issue_dict['issue_confidence'])
        self.assertEqual('test name', issue_dict['issue_text'])