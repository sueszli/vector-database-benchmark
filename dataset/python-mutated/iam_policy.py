"""
.. module: security_monkey.auditors.iam.iam_policy
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Patrick Kelley <pkelley@netflix.com> @monkeysecurity

"""
from security_monkey.auditor import Auditor, Categories
from security_monkey.watchers.iam.managed_policy import ManagedPolicy
from security_monkey import app
import json

class IAMPolicyAuditor(Auditor):

    def __init__(self, accounts=None, debug=False):
        if False:
            for i in range(10):
                print('nop')
        super(IAMPolicyAuditor, self).__init__(accounts=accounts, debug=debug)
        self.iam_policy_keys = ['InlinePolicies$*']

    def load_iam_policies(self, item):
        if False:
            for i in range(10):
                print('nop')
        return self.load_policies(item, self.iam_policy_keys)

    def check_star_privileges(self, item):
        if False:
            for i in range(10):
                print('nop')
        "\n        alert when an IAM Object has a policy allowing '*'.\n        "
        issue = Categories.ADMIN_ACCESS
        notes = Categories.ADMIN_ACCESS_NOTES
        for policy in self.load_iam_policies(item):
            for statement in policy.statements:
                if statement.effect == 'Allow':
                    if '*' in statement.actions:
                        resources = json.dumps(sorted(list(statement.resources)))
                        notes = notes.format(actions='["*"]', resource=resources)
                        self.add_issue(10, issue, item, notes=notes)

    def check_iam_star_privileges(self, item):
        if False:
            while True:
                i = 10
        "\n        alert when an IAM Object has a policy allowing 'iam:*'.\n        "
        issue = Categories.ADMIN_ACCESS
        notes = Categories.ADMIN_ACCESS_NOTES
        for policy in self.load_iam_policies(item):
            for statement in policy.statements:
                if statement.effect == 'Allow':
                    actions = {action.lower() for action in statement.actions}
                    if 'iam:*' in actions:
                        resources = json.dumps(sorted(list(statement.resources)))
                        notes = notes.format(actions='["iam:*"]', resource=resources)
                        self.add_issue(10, issue, item, notes=notes)

    def check_permissions(self, item):
        if False:
            print('Hello World!')
        '\n        Alert when an IAM Object has a policy allowing permission modification.\n        '
        issue = Categories.SENSITIVE_PERMISSIONS
        notes = Categories.SENSITIVE_PERMISSIONS_NOTES_2
        for policy in self.load_iam_policies(item):
            for statement in policy.statements:
                if '*' in statement.actions:
                    continue
                if statement.effect == 'Allow':
                    summary = statement.action_summary()
                    for (service, categories) in list(summary.items()):
                        if 'Permissions' in categories:
                            note = notes.format(service=service, category='Permissions', resource=json.dumps(sorted(list(statement.resources))))
                            self.add_issue(10, issue, item, notes=note)

    def check_mutable_sensitive_services(self, item):
        if False:
            return 10
        '\n        Alert when an IAM Object has DataPlaneMutating permissions for sensitive services.\n        '
        issue = Categories.SENSITIVE_PERMISSIONS
        notes = Categories.SENSITIVE_PERMISSIONS_NOTES_2
        sensitive_services = app.config.get('SENSITIVE_SERVICES', 'ALL')
        if not sensitive_services:
            return
        for policy in self.load_iam_policies(item):
            for statement in policy.statements:
                if '*' in statement.actions:
                    continue
                if statement.effect == 'Allow':
                    summary = statement.action_summary()
                    for (service, categories) in list(summary.items()):
                        if sensitive_services != 'ALL' and service not in sensitive_services:
                            continue
                        if 'DataPlaneMutating' in categories or 'Write' in categories:
                            note = notes.format(service=service, category='DataPlaneWriteAccess', resource=json.dumps(sorted(list(statement.resources))))
                            self.add_issue(1, issue, item, notes=note)

    def check_iam_passrole(self, item):
        if False:
            for i in range(10):
                print('nop')
        "\n        alert when an IAM Object has a policy allowing 'iam:PassRole'.\n        This allows the object to pass any role specified in the resource block to an ec2 instance.\n        "
        issue = Categories.SENSITIVE_PERMISSIONS
        notes = Categories.SENSITIVE_PERMISSIONS_NOTES_1
        for policy in self.load_iam_policies(item):
            for statement in policy.statements:
                if '*' in statement.actions:
                    continue
                if statement.effect == 'Allow':
                    if 'iam:passrole' in statement.actions_expanded:
                        resources = json.dumps(sorted(list(statement.resources)))
                        notes = notes.format(actions='["iam:passrole"]', resource=resources)
                        self.add_issue(10, issue, item, notes=notes)

    def check_notaction(self, item):
        if False:
            i = 10
            return i + 15
        '\n        alert when an IAM Object has a policy containing \'NotAction\'.\n        NotAction combined with an "Effect": "Allow" often provides more privilege\n        than is desired.\n        '
        issue = Categories.STATEMENT_CONSTRUCTION
        notes = Categories.STATEMENT_CONSTRUCTION_NOTES
        for policy in self.load_iam_policies(item):
            for statement in policy.statements:
                if statement.effect == 'Allow':
                    if 'NotAction' in statement.statement:
                        notes = notes.format(construct='["NotAction"]')
                        self.add_issue(10, issue, item, notes=notes)

    def check_notresource(self, item):
        if False:
            i = 10
            return i + 15
        '\n        alert when an IAM Object has a policy containing \'NotResoure\'.\n        NotResource combined with an "Effect": "Allow" often provides more privilege\n        than is desired.\n        '
        issue = Categories.STATEMENT_CONSTRUCTION
        notes = Categories.STATEMENT_CONSTRUCTION_NOTES
        for policy in self.load_iam_policies(item):
            for statement in policy.statements:
                if statement.effect == 'Allow':
                    if 'NotResource' in statement.statement:
                        notes = notes.format(construct='["NotResource"]')
                        self.add_issue(10, issue, item, notes=notes)

    def check_security_group_permissions(self, item):
        if False:
            print('Hello World!')
        '\n        alert when an IAM Object has ec2:AuthorizeSecurityGroupEgress or ec2:AuthorizeSecurityGroupIngress.\n        '
        issue = Categories.SENSITIVE_PERMISSIONS
        notes = Categories.SENSITIVE_PERMISSIONS_NOTES_1
        permissions = {'ec2:authorizesecuritygroupegress', 'ec2:authorizesecuritygroupingress'}
        for policy in self.load_iam_policies(item):
            for statement in policy.statements:
                if '*' in statement.actions:
                    continue
                if statement.effect == 'Allow':
                    permissions = statement.actions_expanded.intersection(permissions)
                    if permissions:
                        actions = json.dumps(sorted(list(permissions)))
                        resources = json.dumps(sorted(list(statement.resources)))
                        note = notes.format(actions=actions, resource=resources)
                        self.add_issue(7, issue, item, notes=note)

    def library_check_attached_managed_policies(self, iam_item, iam_type):
        if False:
            print('Hello World!')
        '\n        alert when an IAM item (group, user or role) is attached to a managed policy with issues\n        '
        mp_items = self.get_auditor_support_items(ManagedPolicy.index, iam_item.account)
        managed_policies = iam_item.config.get('managed_policies', iam_item.config.get('ManagedPolicies'))
        for item_mp in managed_policies or []:
            found = False
            item_mp_arn = item_mp.get('arn', item_mp.get('Arn'))
            for mp_item in mp_items or ([] and (not found)):
                mp_arn = mp_item.config.get('arn', mp_item.config.get('Arn'))
                if mp_arn == item_mp_arn:
                    found = True
            if not found:
                app.logger.error('IAM Managed Policy defined but not found for {}-{}'.format(iam_item.index, iam_item.name))