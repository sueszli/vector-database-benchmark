"""
.. module: security_monkey.auditors.gcp.gce_iam
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Tom Melendez <supertom@google.com> @supertom

"""
from security_monkey.auditor import Auditor
from security_monkey.auditors.gcp.util import make_audit_issue, process_issues
from security_monkey.common.gcp.config import AuditorConfig
from security_monkey.common.gcp.error import AuditIssue
from security_monkey.watchers.gcp.iam.serviceaccount import IAMServiceAccount

class IAMServiceAccountAuditor(Auditor):
    index = IAMServiceAccount.index
    i_am_singular = IAMServiceAccount.i_am_singular
    i_am_plural = IAMServiceAccount.i_am_plural
    gcp_config = AuditorConfig.IAMServiceAccount

    def __init__(self, accounts=None, debug=True):
        if False:
            while True:
                i = 10
        super(IAMServiceAccountAuditor, self).__init__(accounts=accounts, debug=debug)

    def _max_keys(self, key_count, error_cat='SA'):
        if False:
            while True:
                i = 10
        '\n        Alert when a service account has too many keys.\n\n        return: [list of AuditIssues]\n        '
        errors = []
        if key_count > self.gcp_config.MAX_SERVICEACCOUNT_KEYS:
            ae = make_audit_issue(error_cat, 'MAX', 'KEYS')
            ae.notes = 'Too Many Keys (count: %s, max: %s)' % (key_count, self.gcp_config.MAX_SERVICEACCOUNT_KEYS)
            errors.append(ae)
        return errors

    def _actor_role(self, policies, error_cat='SA'):
        if False:
            i = 10
            return i + 15
        '\n        Determine if a serviceaccount actor is specified.\n\n        return: [list of AuditIssues]\n        '
        errors = []
        for policy in policies:
            role = policy.get('Role')
            if role and role == 'iam.serviceAccountActor':
                ae = make_audit_issue(error_cat, 'POLICY', 'ROLE', 'ACTOR')
                errors.append(ae)
        return errors

    def inspect_serviceaccount(self, item):
        if False:
            return 10
        '\n        Driver for ServiceAccount. Calls helpers as needed.\n\n        return: (bool, [list of AuditIssues])\n        '
        errors = []
        err = self._max_keys(item.config.get('keys'))
        errors.extend(err) if err else None
        policies = item.config.get('policy')
        if policies:
            err = self._actor_role(policies)
            errors.extend(err) if err else None
        if errors:
            return (False, errors)
        return (True, None)

    def check_serviceaccount(self, item):
        if False:
            i = 10
            return i + 15
        (ok, errors) = self.inspect_serviceaccount(item)
        process_issues(self, ok, errors, item)