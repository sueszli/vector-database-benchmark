"""
.. module: security_monkey.accounts.aws_account
    :platform: Unix
    :synopsis: Manages generic AWS account.


.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.account_manager import AccountManager, CustomFieldConfig

class AWSAccountManager(AccountManager):
    account_type = 'AWS'
    identifier_label = 'Number'
    identifier_tool_tip = 'Enter the AWS account number, if you have it. (12 digits)'
    s3_name_label = '[DEPRECATED -- use canonical id] The S3 Name is the way AWS presents the account in an ACL policy.  This is often times the first part of the email address that was used to create the Amazon account.  (myaccount@example.com may be represented as myaccount\\).  If you see S3 issues appear for unknown cross account access, you may need to update the S3 Name.'
    s3_canonical_id = "The Canonical ID is the way AWS presents the account in an ACL policy.  It is a unique set of characters that is tied to an AWS account.  If you see S3 issues appear for unknown cross account access, you may need to update the canonical ID.  A manager.py command has been included that can fetch this for you automatically (fetch_aws_canonical_ids), since it requires a 'list_buckets' API call against AWS to obtain."
    role_name_label = "Optional custom role name, otherwise the default 'SecurityMonkey' is used. When deploying roles via CloudFormation, this is the Physical ID of the generated IAM::ROLE."
    external_id_label = 'Optional custom external id. See https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html'
    custom_field_configs = [CustomFieldConfig('canonical_id', 'Canonical ID', True, s3_canonical_id), CustomFieldConfig('s3_name', 'S3 Name', True, s3_name_label), CustomFieldConfig('role_name', 'Role Name', True, role_name_label), CustomFieldConfig('external_id', 'External Id', True, external_id_label)]

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(AWSAccountManager, self).__init__()

    def sanitize_account_identifier(self, identifier):
        if False:
            for i in range(10):
                print('nop')
        'AWS identifer sanitization will strip and remove any hyphens.\n\n        Returns:\n            stripped identifier with hyphens removed\n        '
        return identifier.replace('-', '').strip()