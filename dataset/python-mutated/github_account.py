"""
.. module: security_monkey.accounts.github_account
    :platform: Unix
    :synopsis: Manages GitHub Organizations.


.. version:: $$VERSION$$
.. moduleauthor:: Mike Grima <mgrima@netflix.com>


"""
from security_monkey.account_manager import AccountManager, CustomFieldConfig

class GitHubAccountManager(AccountManager):
    account_type = 'GitHub'
    identifier_label = 'Organization Name'
    identifier_tool_tip = 'Enter the GitHub Organization Name'
    access_token_tool_tip = 'Enter the path to the file that contains the GitHub personal access token.'
    custom_field_configs = [CustomFieldConfig('access_token_file', 'Personal Access Token', True, access_token_tool_tip)]

    def __init__(self):
        if False:
            while True:
                i = 10
        super(GitHubAccountManager, self).__init__()