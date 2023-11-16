from odoo.addons.project_issue.tests.common import TestIssueUsers

class TestIssueProcess(TestIssueUsers):

    def test_issue_process(self):
        if False:
            for i in range(10):
                print('nop')
        vals = {'email_from': 'support@mycompany.com', 'email_to': 'Robert_Adersen@yahoo.com', 'subject': 'We need more details regarding your issue in HR module', 'body_html': '\n                    <p>\n                        Hello Mr. Adersen,\n                    </p>\n                    <p>\n                        We need more details about your issue in the HR module. Could you please\n                        send us complete details about the error eg. error message, traceback\n                        or what operations you were doing when you the error occured ?\n                    </p>\n                    <p>\n                        Thank You.\n                    </p>\n                    <pre>\n--\nYourCompany\ninfo@yourcompany.example.com\n+1 555 123 8069\n                    </pre>\n                '}
        crm_bug_id = self.ref('project_issue.crm_case_buginaccountsmodule0')
        mail = self.env['mail.mail'].with_context(active_model='project.issue', active_id=crm_bug_id, active_ids=[crm_bug_id]).create(vals)
        mail.send()