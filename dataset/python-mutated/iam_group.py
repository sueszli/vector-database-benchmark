"""
.. module: security_monkey.auditors.iam.iam_group
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Patrick Kelley <pkelley@netflix.com> @monkeysecurity

"""
from security_monkey.watchers.iam.iam_group import IAMGroup
from security_monkey.auditors.iam.iam_policy import IAMPolicyAuditor
from security_monkey.watchers.iam.managed_policy import ManagedPolicy

class IAMGroupAuditor(IAMPolicyAuditor):
    index = IAMGroup.index
    i_am_singular = IAMGroup.i_am_singular
    i_am_plural = IAMGroup.i_am_plural
    support_auditor_indexes = [ManagedPolicy.index]

    def __init__(self, accounts=None, debug=False):
        if False:
            print('Hello World!')
        super(IAMGroupAuditor, self).__init__(accounts=accounts, debug=debug)
        self.iam_policy_keys = ['grouppolicies$*']

    def check_attached_managed_policies(self, iamgroup_item):
        if False:
            return 10
        '\n        alert when an IAM Group is attached to a managed policy with issues\n        '
        self.library_check_attached_managed_policies(iamgroup_item, 'group')