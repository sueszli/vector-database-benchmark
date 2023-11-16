"""
.. module: security_monkey.auditors.iam.iam_user
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Patrick Kelley <pkelley@netflix.com> @monkeysecurity

"""
import datetime
from dateutil import parser
from dateutil import tz
from security_monkey.auditor import Categories
from security_monkey.watchers.iam.iam_user import IAMUser
from security_monkey.auditors.iam.iam_policy import IAMPolicyAuditor
from security_monkey.watchers.iam.managed_policy import ManagedPolicy

class IAMUserAuditor(IAMPolicyAuditor):
    index = IAMUser.index
    i_am_singular = IAMUser.i_am_singular
    i_am_plural = IAMUser.i_am_plural
    support_auditor_indexes = [ManagedPolicy.index]

    def __init__(self, accounts=None, debug=False):
        if False:
            i = 10
            return i + 15
        super(IAMUserAuditor, self).__init__(accounts=accounts, debug=debug)
        self.iam_policy_keys = ['InlinePolicies$*']

    def prep_for_audit(self):
        if False:
            return 10
        '\n        Prepare for the audit by calculating 90 days ago.\n        This is used to check if access keys have been rotated.\n        '
        super(IAMUserAuditor, self).prep_for_audit()
        now = datetime.datetime.now()
        then = now - datetime.timedelta(days=90)
        self.ninety_days_ago = then.replace(tzinfo=tz.gettz('UTC'))

    def check_active_access_keys(self, item):
        if False:
            print('Hello World!')
        '\n        alert when an IAM User has an active access key.\n        score: 1\n        '
        issue = Categories.INFORMATIONAL
        notes = Categories.INFORMATIONAL_NOTES
        akeys = item.config.get('AccessKeys', {})
        for akey in akeys:
            if 'Status' in akey:
                if akey['Status'] == 'Active':
                    note = notes.format(description='Active Accesskey', specific=' [{}]'.format(akey['AccessKeyId']))
                    self.add_issue(1, issue, item, notes=note)

    def check_inactive_access_keys(self, item):
        if False:
            i = 10
            return i + 15
        '\n        alert when an IAM User has an inactive access key.\n        score: 0\n        '
        issue = Categories.INFORMATIONAL
        notes = Categories.INFORMATIONAL_NOTES
        akeys = item.config.get('AccessKeys', {})
        for akey in akeys:
            if 'Status' in akey:
                if akey['Status'] != 'Active':
                    description = 'Inactive Accesskey'
                    specific = ' [{}]'.format(akey['AccessKeyId'])
                    note = notes.format(description=description, specific=specific)
                    self.add_issue(0, issue, item, notes=note)

    def check_access_key_rotation(self, item):
        if False:
            print('Hello World!')
        '\n        alert when an IAM User has an active access key created more than 90 days go.\n        '
        issue = Categories.ROTATION
        notes = Categories.ROTATION_NOTES
        requirement = '> 90 days ago'
        akeys = item.config.get('AccessKeys', {})
        for akey in akeys:
            if 'Status' in akey:
                if akey['Status'] == 'Active':
                    create_date = akey['CreateDate']
                    create_date = parser.parse(create_date)
                    if create_date < self.ninety_days_ago:
                        note = notes.format(what='Active Accesskey [{key}]'.format(key=akey['AccessKeyId']), requirement=requirement, date=akey['CreateDate'])
                        self.add_issue(1, issue, item, notes=note)

    def check_access_key_last_used(self, item):
        if False:
            print('Hello World!')
        "\n        alert if an active access key hasn't been used in 90 days\n        "
        issue = Categories.UNUSED
        notes = Categories.UNUSED_NOTES
        requirement = '> 90 days ago'
        akeys = item.config.get('AccessKeys', {})
        for akey in akeys:
            if 'Status' in akey:
                if akey['Status'] == 'Active':
                    last_used_str = akey.get('LastUsedDate') or akey.get('CreateDate')
                    last_used_date = parser.parse(last_used_str)
                    if last_used_date < self.ninety_days_ago:
                        note = notes.format(what='Active Accesskey [{key}]'.format(key=akey['AccessKeyId']), requirement=requirement, date=last_used_str)
                        self.add_issue(1, issue, item, notes=note)

    def check_no_mfa(self, item):
        if False:
            while True:
                i = 10
        '\n        alert when an IAM user has a login profile and no MFA devices.\n        This means a human account which could be better protected with 2FA.\n        '
        issue = Categories.INSECURE_CONFIGURATION
        notes = Categories.INSECURE_CONFIGURATION_NOTES
        notes = notes.format(description='User with password login and no MFA devices')
        user_mfas = item.config.get('MfaDevices', {})
        login_profile = item.config.get('LoginProfile', {})
        if login_profile and (not user_mfas):
            self.add_issue(1, issue, item, notes=notes)

    def check_loginprofile_plus_akeys(self, item):
        if False:
            while True:
                i = 10
        '\n        alert when an IAM user has a login profile and API access via access keys.\n        An account should be used Either for API access OR for console access, but maybe not both.\n        '
        if not item.config.get('LoginProfile', None):
            return
        issue = Categories.INFORMATIONAL
        notes = Categories.INFORMATIONAL_NOTES
        notes = notes.format(description='User with password login and API access', specific='')
        akeys = item.config.get('AccessKeys', {})
        for akey in akeys:
            if 'Status' in akey and akey['Status'] == 'Active':
                self.add_issue(1, issue, item, notes)
                return

    def check_attached_managed_policies(self, iamuser_item):
        if False:
            while True:
                i = 10
        '\n        alert when an IAM Role is attached to a managed policy with issues\n        '
        self.library_check_attached_managed_policies(iamuser_item, 'user')