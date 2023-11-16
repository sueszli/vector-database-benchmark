"""
.. module: security_monkey.tests.auditors.test_managed_policy
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>

"""
from security_monkey.tests import SecurityMonkeyTestCase
from security_monkey.auditors.iam.managed_policy import ManagedPolicyAuditor
from security_monkey.watchers.iam.managed_policy import ManagedPolicyItem
from security_monkey import ARN_PREFIX
FULL_ADMIN_POLICY_BARE = '\n{\n    "Statement":    {\n        "Effect": "Allow",\n        "Action": "*"\n    }\n}\n'

class ManagedPolicyAuditorTestCase(SecurityMonkeyTestCase):

    def test_issue_on_non_aws_policy(self):
        if False:
            return 10
        import json
        config = {'policy': json.loads(FULL_ADMIN_POLICY_BARE), 'arn': ARN_PREFIX + ':iam::123456789:policy/TEST', 'attached_users': [], 'attached_roles': [], 'attached_groups': []}
        auditor = ManagedPolicyAuditor(accounts=['unittest'])
        policyobj = ManagedPolicyItem(account='TEST_ACCOUNT', name='policy_test', config=config)
        self.assertIs(len(policyobj.audit_issues), 0, 'Managed Policy should have 0 alert but has {}'.format(len(policyobj.audit_issues)))
        auditor.check_star_privileges(policyobj)
        self.assertIs(len(policyobj.audit_issues), 1, 'Managed Policy should have 1 alert but has {}'.format(len(policyobj.audit_issues)))

    def test_issue_on_aws_policy_no_attachments(self):
        if False:
            for i in range(10):
                print('nop')
        import json
        config = {'policy': json.loads(FULL_ADMIN_POLICY_BARE), 'arn': ARN_PREFIX + ':iam::aws:policy/TEST', 'attached_users': [], 'attached_roles': [], 'attached_groups': []}
        auditor = ManagedPolicyAuditor(accounts=['unittest'])
        policyobj = ManagedPolicyItem(account='TEST_ACCOUNT', name='policy_test', config=config)
        self.assertIs(len(policyobj.audit_issues), 0, 'Managed Policy should have 0 alert but has {}'.format(len(policyobj.audit_issues)))
        auditor.check_star_privileges(policyobj)
        self.assertIs(len(policyobj.audit_issues), 0, 'Managed Policy should have 0 alerts but has {}'.format(len(policyobj.audit_issues)))

    def test_issue_on_aws_policy_with_attachment(self):
        if False:
            for i in range(10):
                print('nop')
        import json
        config = {'policy': json.loads(FULL_ADMIN_POLICY_BARE), 'arn': ARN_PREFIX + ':iam::aws:policy/TEST', 'attached_users': [], 'attached_roles': [ARN_PREFIX + ':iam::123456789:role/TEST'], 'attached_groups': []}
        auditor = ManagedPolicyAuditor(accounts=['unittest'])
        policyobj = ManagedPolicyItem(account='TEST_ACCOUNT', name='policy_test', config=config)
        self.assertIs(len(policyobj.audit_issues), 0, 'Managed Policy should have 0 alert but has {}'.format(len(policyobj.audit_issues)))
        auditor.check_star_privileges(policyobj)
        self.assertIs(len(policyobj.audit_issues), 1, 'Managed Policy should have 1 alert but has {}'.format(len(policyobj.audit_issues)))