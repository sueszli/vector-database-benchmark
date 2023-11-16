import re
from ..compat import validate_policytype
from . import tags_or_list
Active = 'Active'
Inactive = 'Inactive'

def iam_group_name(group_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: Group.GroupName\n    '
    if len(group_name) > 128:
        raise ValueError('IAM Role Name may not exceed 128 characters')
    iam_names(group_name)
    return group_name

def iam_names(b):
    if False:
        while True:
            i = 10
    iam_name_re = re.compile('^[a-zA-Z0-9_\\.\\+\\=\\@\\-\\,]+$')
    if iam_name_re.match(b):
        return b
    else:
        raise ValueError('%s is not a valid iam name' % b)

def iam_path(path):
    if False:
        print('Hello World!')
    '\n    Property: Group.Path\n    Property: InstanceProfile.Path\n    Property: ManagedPolicy.Path\n    Property: Role.Path\n    Property: User.Path\n    '
    if len(path) > 512:
        raise ValueError('IAM path %s may not exceed 512 characters', path)
    iam_path_re = re.compile('^\\/.*\\/$|^\\/$')
    if not iam_path_re.match(path):
        raise ValueError('%s is not a valid iam path name' % path)
    return path

def iam_role_name(role_name):
    if False:
        return 10
    '\n    Property: Role.RoleName\n    '
    if len(role_name) > 64:
        raise ValueError('IAM Role Name may not exceed 64 characters')
    iam_names(role_name)
    return role_name

def iam_user_name(user_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: User.UserName\n    '
    if not user_name:
        raise ValueError("AWS::IAM::User property 'UserName' may not be empty")
    if len(user_name) > 64:
        raise ValueError("AWS::IAM::User property 'UserName' may not exceed 64 characters")
    iam_user_name_re = re.compile('^[\\w+=,.@-]+$')
    if iam_user_name_re.match(user_name):
        return user_name
    else:
        raise ValueError("%s is not a valid value for AWS::IAM::User property 'UserName'", user_name)

def policytypes(policy):
    if False:
        print('Hello World!')
    '\n    Property: ManagedPolicy.PolicyDocument\n    Property: Policy.PolicyDocument\n    Property: PolicyType.PolicyDocument\n    Property: Role.AssumeRolePolicyDocument\n    '
    return validate_policytype(policy)

def status(status):
    if False:
        return 10
    '\n    Property: AccessKey.Status\n    '
    valid_statuses = [Active, Inactive]
    if status not in valid_statuses:
        raise ValueError('Status needs to be one of %r' % valid_statuses)
    return status

def validate_tags_or_list(x):
    if False:
        while True:
            i = 10
    '\n    Property: Role.Tags\n    '
    return tags_or_list(x)