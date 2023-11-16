from security_monkey.tests import SecurityMonkeyTestCase
'\n.. module: security_monkey.tests.auditors.gcp.iam.test_serviceaccount\n    :platform: Unix\n\n.. version:: $$VERSION$$\n.. moduleauthor::  Tom Melendez <supertom@google.com> @supertom\n'
POLICY_WITH_ACTOR_LIST = [{'Members': ['user:test-user@gmail.com'], 'Role': 'roles/iam.serviceAccountActor'}]
POLICY_NO_ACTOR_LIST = [{'Members': ['user:test-user@gmail.com'], 'Role': 'roles/viewer'}]

class IAMServiceAccountTestCase(SecurityMonkeyTestCase):

    def test__max_keys(self):
        if False:
            i = 10
            return i + 15
        from security_monkey.auditors.gcp.iam.serviceaccount import IAMServiceAccountAuditor
        auditor = IAMServiceAccountAuditor(accounts=['unittest'])
        auditor.gcp_config.MAX_SERVICEACCOUNT_KEYS = 1
        actual = auditor._max_keys(2)
        self.assertTrue(isinstance(actual, list))
        actual = auditor._max_keys(1)
        self.assertFalse(actual)

    def test__actor_role(self):
        if False:
            for i in range(10):
                print('nop')
        from security_monkey.auditors.gcp.iam.serviceaccount import IAMServiceAccountAuditor
        auditor = IAMServiceAccountAuditor(accounts=['unittest'])
        auditor.gcp_config.MAX_SERVICEACCOUNT_KEYS = 1
        actual = auditor._actor_role(POLICY_WITH_ACTOR_LIST)
        self.assertTrue(isinstance(actual, list))
        actual = auditor._actor_role(POLICY_NO_ACTOR_LIST)
        self.assertFalse(actual)