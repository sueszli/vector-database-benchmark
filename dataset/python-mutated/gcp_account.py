"""
.. module: security_monkey.accounts.gcp_account
    :platform: Unix
    :synopsis: Manages generic GCP account.


.. version:: $$VERSION$$
.. moduleauthor:: Tom Melendez (@supertom) <supertom@google.com>


"""
from security_monkey.account_manager import AccountManager, CustomFieldConfig
from security_monkey.datastore import Account

class GCPAccountManager(AccountManager):
    account_type = 'GCP'
    identifier_label = 'Project ID'
    identifier_tool_tip = 'Enter the GCP Project ID.'
    creds_file_tool_tip = 'Enter the path on disk to the credentials file.'
    custom_field_configs = [CustomFieldConfig('creds_file', 'Credentials File', True, creds_file_tool_tip)]

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(GCPAccountManager, self).__init__()

    def lookup_account_by_identifier(self, identifier):
        if False:
            while True:
                i = 10
        '\n        Overrides the lookup to also check the number for backwards compatibility\n        '
        account = super(GCPAccountManager, self).lookup_account_by_identifier(identifier)
        return account

    def _populate_account(self, account, account_type, name, active, third_party, notes, identifier, custom_fields=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        # TODO(supertom): look into this.\n        Overrides create and update to also save the number, s3_name and role_name\n        for backwards compatibility\n        '
        account = super(GCPAccountManager, self)._populate_account(account, account_type, name, active, third_party, notes, identifier, custom_fields)
        return account