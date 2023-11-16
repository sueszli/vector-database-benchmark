"""
Purpose

Shows how to use AWS SDK for Python (Boto3) to create an IAM user, assume a role,
and perform AWS actions.

1. Create a user who has no permissions.
2. Create a role that grants permission to list Amazon S3 buckets for the account.
3. Add a policy to let the user assume the role.
4. Assume the role and list Amazon S3 buckets using temporary credentials.
5. Delete the policy, role, and user.
"""
import json
import sys
import time
from uuid import uuid4
import boto3
from botocore.exceptions import ClientError

def progress_bar(seconds):
    if False:
        for i in range(10):
            print('nop')
    'Shows a simple progress bar in the command window.'
    for _ in range(seconds):
        time.sleep(1)
        print('.', end='')
        sys.stdout.flush()
    print()

def setup(iam_resource):
    if False:
        i = 10
        return i + 15
    '\n    Creates a new user with no permissions.\n    Creates an access key pair for the user.\n    Creates a role with a policy that lets the user assume the role.\n    Creates a policy that allows listing Amazon S3 buckets.\n    Attaches the policy to the role.\n    Creates an inline policy for the user that lets the user assume the role.\n\n    :param iam_resource: A Boto3 AWS Identity and Access Management (IAM) resource\n                         that has permissions to create users, roles, and policies\n                         in the account.\n    :return: The newly created user, user key, and role.\n    '
    try:
        user = iam_resource.create_user(UserName=f'demo-user-{uuid4()}')
        print(f'Created user {user.name}.')
    except ClientError as error:
        print(f"Couldn't create a user for the demo. Here's why: {error.response['Error']['Message']}")
        raise
    try:
        user_key = user.create_access_key_pair()
        print(f'Created access key pair for user.')
    except ClientError as error:
        print(f"Couldn't create access keys for user {user.name}. Here's why: {error.response['Error']['Message']}")
        raise
    print(f'Wait for user to be ready.', end='')
    progress_bar(10)
    try:
        role = iam_resource.create_role(RoleName=f'demo-role-{uuid4()}', AssumeRolePolicyDocument=json.dumps({'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'AWS': user.arn}, 'Action': 'sts:AssumeRole'}]}))
        print(f'Created role {role.name}.')
    except ClientError as error:
        print(f"Couldn't create a role for the demo. Here's why: {error.response['Error']['Message']}")
        raise
    try:
        policy = iam_resource.create_policy(PolicyName=f'demo-policy-{uuid4()}', PolicyDocument=json.dumps({'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 's3:ListAllMyBuckets', 'Resource': 'arn:aws:s3:::*'}]}))
        role.attach_policy(PolicyArn=policy.arn)
        print(f'Created policy {policy.policy_name} and attached it to the role.')
    except ClientError as error:
        print(f"Couldn't create a policy and attach it to role {role.name}. Here's why: {error.response['Error']['Message']}")
        raise
    try:
        user.create_policy(PolicyName=f'demo-user-policy-{uuid4()}', PolicyDocument=json.dumps({'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'sts:AssumeRole', 'Resource': role.arn}]}))
        print(f'Created an inline policy for {user.name} that lets the user assume the role.')
    except ClientError as error:
        print(f"Couldn't create an inline policy for user {user.name}. Here's why: {error.response['Error']['Message']}")
        raise
    print('Give AWS time to propagate these new resources and connections.', end='')
    progress_bar(10)
    return (user, user_key, role)

def show_access_denied_without_role(user_key):
    if False:
        while True:
            i = 10
    '\n    Shows that listing buckets without first assuming the role is not allowed.\n\n    :param user_key: The key of the user created during setup. This user does not\n                     have permission to list buckets in the account.\n    '
    print(f'Try to list buckets without first assuming the role.')
    s3_denied_resource = boto3.resource('s3', aws_access_key_id=user_key.id, aws_secret_access_key=user_key.secret)
    try:
        for bucket in s3_denied_resource.buckets.all():
            print(bucket.name)
        raise RuntimeError('Expected to get AccessDenied error when listing buckets!')
    except ClientError as error:
        if error.response['Error']['Code'] == 'AccessDenied':
            print('Attempt to list buckets with no permissions: AccessDenied.')
        else:
            raise

def list_buckets_from_assumed_role(user_key, assume_role_arn, session_name):
    if False:
        return 10
    "\n    Assumes a role that grants permission to list the Amazon S3 buckets in the account.\n    Uses the temporary credentials from the role to list the buckets that are owned\n    by the assumed role's account.\n\n    :param user_key: The access key of a user that has permission to assume the role.\n    :param assume_role_arn: The Amazon Resource Name (ARN) of the role that\n                            grants access to list the other account's buckets.\n    :param session_name: The name of the STS session.\n    "
    sts_client = boto3.client('sts', aws_access_key_id=user_key.id, aws_secret_access_key=user_key.secret)
    try:
        response = sts_client.assume_role(RoleArn=assume_role_arn, RoleSessionName=session_name)
        temp_credentials = response['Credentials']
        print(f'Assumed role {assume_role_arn} and got temporary credentials.')
    except ClientError as error:
        print(f"Couldn't assume role {assume_role_arn}. Here's why: {error.response['Error']['Message']}")
        raise
    s3_resource = boto3.resource('s3', aws_access_key_id=temp_credentials['AccessKeyId'], aws_secret_access_key=temp_credentials['SecretAccessKey'], aws_session_token=temp_credentials['SessionToken'])
    print(f"Listing buckets for the assumed role's account:")
    try:
        for bucket in s3_resource.buckets.all():
            print(bucket.name)
    except ClientError as error:
        print(f"Couldn't list buckets for the account. Here's why: {error.response['Error']['Message']}")
        raise

def teardown(user, role):
    if False:
        print('Hello World!')
    '\n    Removes all resources created during setup.\n\n    :param user: The demo user.\n    :param role: The demo role.\n    '
    try:
        for attached in role.attached_policies.all():
            policy_name = attached.policy_name
            role.detach_policy(PolicyArn=attached.arn)
            attached.delete()
            print(f'Detached and deleted {policy_name}.')
        role.delete()
        print(f'Deleted {role.name}.')
    except ClientError as error:
        print(f"Couldn't detach policy, delete policy, or delete role. Here's why: {error.response['Error']['Message']}")
        raise
    try:
        for user_pol in user.policies.all():
            user_pol.delete()
            print('Deleted inline user policy.')
        for key in user.access_keys.all():
            key.delete()
            print("Deleted user's access key.")
        user.delete()
        print(f'Deleted {user.name}.')
    except ClientError as error:
        print(f"Couldn't delete user policy or delete user. Here's why: {error.response['Error']['Message']}")

def usage_demo():
    if False:
        i = 10
        return i + 15
    'Drives the demonstration.'
    print('-' * 88)
    print(f'Welcome to the IAM create user and assume role demo.')
    print('-' * 88)
    iam_resource = boto3.resource('iam')
    user = None
    role = None
    try:
        (user, user_key, role) = setup(iam_resource)
        print(f'Created {user.name} and {role.name}.')
        show_access_denied_without_role(user_key)
        list_buckets_from_assumed_role(user_key, role.arn, 'AssumeRoleDemoSession')
    except Exception:
        print('Something went wrong!')
    finally:
        if user is not None and role is not None:
            teardown(user, role)
        print('Thanks for watching!')
if __name__ == '__main__':
    usage_demo()