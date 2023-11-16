"""
.. module: security_monkey.auditors.iam.iam_role
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Patrick Kelley <pkelley@netflix.com> @monkeysecurity

"""
from security_monkey.watchers.iam.iam_role import IAMRole
from security_monkey.auditors.iam.iam_policy import IAMPolicyAuditor
from security_monkey.auditors.resource_policy_auditor import ResourcePolicyAuditor
from security_monkey.watchers.iam.managed_policy import ManagedPolicy

class IAMRoleAuditor(IAMPolicyAuditor, ResourcePolicyAuditor):
    index = IAMRole.index
    i_am_singular = IAMRole.i_am_singular
    i_am_plural = IAMRole.i_am_plural
    support_auditor_indexes = [ManagedPolicy.index]

    def __init__(self, accounts=None, debug=False):
        if False:
            print('Hello World!')
        super(IAMRoleAuditor, self).__init__(accounts=accounts, debug=debug)
        self.policy_keys = ['AssumeRolePolicyDocument']
        self.iam_policy_keys = ['InlinePolicies$*']

    def check_attached_managed_policies(self, iamrole_item):
        if False:
            i = 10
            return i + 15
        '\n        alert when an IAM Role is attached to a managed policy with issues\n        '
        self.library_check_attached_managed_policies(iamrole_item, 'role')