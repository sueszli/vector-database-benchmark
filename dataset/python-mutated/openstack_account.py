"""
.. module: security_monkey.accounts.openstack_account
    :platform: Unix
    :synopsis: Manages generic OpenStack account.


.. version:: $$VERSION$$
.. moduleauthor:: Michael Stair <mstair@att.com>


"""
from security_monkey.account_manager import AccountManager, CustomFieldConfig
from security_monkey.datastore import Account

class OpenStackAccountManager(AccountManager):
    account_type = 'OpenStack'
    identifier_label = 'Cloud Name'
    identifier_tool_tip = 'OpenStack Cloud Name. Cloud configuration to load from clouds.yaml file'
    cloudsyaml_tool_tip = 'Path on disk to clouds.yaml file'
    custom_field_configs = [CustomFieldConfig('cloudsyaml_file', 'OpenStack clouds.yaml file', True, cloudsyaml_tool_tip)]

    def __init__(self):
        if False:
            while True:
                i = 10
        super(OpenStackAccountManager, self).__init__()