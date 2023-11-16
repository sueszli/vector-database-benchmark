"""
Purpose

Shows how to assume a role that requires a multi-factor authentication (MFA) token,
using AWS Security Token Service (STS) credentials.
"""
import json
import os
import time
import sys
import webbrowser
import boto3
from botocore.exceptions import ClientError

def progress_bar(seconds):
    if False:
        return 10
    'Shows a simple progress bar in the command window.'
    for _ in range(seconds):
        time.sleep(1)
        print('.', end='')
        sys.stdout.flush()
    print()

def unique_name(base_name):
    if False:
        while True:
            i = 10
    return f'demo-assume-role-{base_name}-{time.time_ns()}'

def setup(iam_resource):
    if False:
        while True:
            i = 10
    '\n    Creates a new user with no permissions.\n    Creates a new virtual MFA device.\n    Displays the QR code to seed the device.\n    Asks for two codes from the MFA device.\n    Registers the MFA device for the user.\n    Creates an access key pair for the user.\n    Creates a role with a policy that lets the user assume the role and requires MFA.\n    Creates a policy that allows listing Amazon S3 buckets.\n    Attaches the policy to the role.\n    Creates an inline policy for the user that lets the user assume the role.\n\n    For demonstration purposes, the user is created in the same account as the role,\n    but in practice the user would likely be from another account.\n\n    Any MFA device that can scan a QR code will work with this demonstration.\n    Common choices are mobile apps like LastPass Authenticator,\n    Microsoft Authenticator, or Google Authenticator.\n\n    :param iam_resource: A Boto3 AWS Identity and Access Management (IAM) resource\n                         that has permissions to create users, roles, and policies\n                         in the account.\n    :return: The newly created user, user key, virtual MFA device, and role.\n    '
    user = iam_resource.create_user(UserName=unique_name('user'))
    print(f'Created user {user.name}.')
    virtual_mfa_device = iam_resource.create_virtual_mfa_device(VirtualMFADeviceName=unique_name('mfa'))
    print(f'Created virtual MFA device {virtual_mfa_device.serial_number}')
    print(f'Showing the QR code for the device. Scan this in the MFA app of your choice.')
    with open('qr.png', 'wb') as qr_file:
        qr_file.write(virtual_mfa_device.qr_code_png)
    webbrowser.open(qr_file.name)
    print(f'Enter two consecutive code from your MFA device.')
    mfa_code_1 = input('Enter the first code: ')
    mfa_code_2 = input('Enter the second code: ')
    user.enable_mfa(SerialNumber=virtual_mfa_device.serial_number, AuthenticationCode1=mfa_code_1, AuthenticationCode2=mfa_code_2)
    os.remove(qr_file.name)
    print(f'MFA device is registered with the user.')
    user_key = user.create_access_key_pair()
    print(f'Created access key pair for user.')
    print(f'Wait for user to be ready.', end='')
    progress_bar(10)
    role = iam_resource.create_role(RoleName=unique_name('role'), AssumeRolePolicyDocument=json.dumps({'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'AWS': user.arn}, 'Action': 'sts:AssumeRole', 'Condition': {'Bool': {'aws:MultiFactorAuthPresent': True}}}]}))
    print(f'Created role {role.name} that requires MFA.')
    policy = iam_resource.create_policy(PolicyName=unique_name('policy'), PolicyDocument=json.dumps({'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 's3:ListAllMyBuckets', 'Resource': 'arn:aws:s3:::*'}]}))
    role.attach_policy(PolicyArn=policy.arn)
    print(f'Created policy {policy.policy_name} and attached it to the role.')
    user.create_policy(PolicyName=unique_name('user-policy'), PolicyDocument=json.dumps({'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'sts:AssumeRole', 'Resource': role.arn}]}))
    print(f'Created an inline policy for {user.name} that lets the user assume the role.')
    print('Give AWS time to propagate these new resources and connections.', end='')
    progress_bar(10)
    return (user, user_key, virtual_mfa_device, role)

def try_to_assume_role_without_mfa(assume_role_arn, session_name, sts_client):
    if False:
        for i in range(10):
            print('nop')
    '\n    Shows that attempting to assume the role without sending MFA credentials results\n    in an AccessDenied error.\n\n    :param assume_role_arn: The Amazon Resource Name (ARN) of the role to assume.\n    :param session_name: The name of the STS session.\n    :param sts_client: A Boto3 STS instance that has permission to assume the role.\n    '
    print(f'Trying to assume the role without sending MFA credentials...')
    try:
        sts_client.assume_role(RoleArn=assume_role_arn, RoleSessionName=session_name)
        raise RuntimeError('Expected AccessDenied error.')
    except ClientError as error:
        if error.response['Error']['Code'] == 'AccessDenied':
            print('Got AccessDenied.')
        else:
            raise

def list_buckets_from_assumed_role_with_mfa(assume_role_arn, session_name, mfa_serial_number, mfa_totp, sts_client):
    if False:
        for i in range(10):
            print('nop')
    "\n    Assumes a role from another account and uses the temporary credentials from\n    that role to list the Amazon S3 buckets that are owned by the other account.\n    Requires an MFA device serial number and token.\n\n    The assumed role must grant permission to list the buckets in the other account.\n\n    :param assume_role_arn: The Amazon Resource Name (ARN) of the role that\n                            grants access to list the other account's buckets.\n    :param session_name: The name of the STS session.\n    :param mfa_serial_number: The serial number of the MFA device. For a virtual MFA\n                              device, this is an ARN.\n    :param mfa_totp: A time-based, one-time password issued by the MFA device.\n    :param sts_client: A Boto3 STS instance that has permission to assume the role.\n    "
    response = sts_client.assume_role(RoleArn=assume_role_arn, RoleSessionName=session_name, SerialNumber=mfa_serial_number, TokenCode=mfa_totp)
    temp_credentials = response['Credentials']
    print(f'Assumed role {assume_role_arn} and got temporary credentials.')
    s3_resource = boto3.resource('s3', aws_access_key_id=temp_credentials['AccessKeyId'], aws_secret_access_key=temp_credentials['SecretAccessKey'], aws_session_token=temp_credentials['SessionToken'])
    print(f"Listing buckets for the assumed role's account:")
    for bucket in s3_resource.buckets.all():
        print(bucket.name)

def teardown(user, virtual_mfa_device, role):
    if False:
        i = 10
        return i + 15
    '\n    Removes all resources created during setup.\n\n    :param user: The demo user.\n    :param role: The demo role.\n    '
    for attached in role.attached_policies.all():
        policy_name = attached.policy_name
        role.detach_policy(PolicyArn=attached.arn)
        attached.delete()
        print(f'Detached and deleted {policy_name}.')
    role.delete()
    print(f'Deleted {role.name}.')
    for user_pol in user.policies.all():
        user_pol.delete()
        print('Deleted inline user policy.')
    for key in user.access_keys.all():
        key.delete()
        print("Deleted user's access key.")
    for mfa in user.mfa_devices.all():
        mfa.disassociate()
    virtual_mfa_device.delete()
    user.delete()
    print(f'Deleted {user.name}.')

def usage_demo():
    if False:
        return 10
    'Drives the demonstration.'
    print('-' * 88)
    print(f'Welcome to the AWS Security Token Service assume role demo, starring multi-factor authentication (MFA)!')
    print('-' * 88)
    iam_resource = boto3.resource('iam')
    (user, user_key, virtual_mfa_device, role) = setup(iam_resource)
    print(f'Created {user.name} and {role.name}.')
    try:
        sts_client = boto3.client('sts', aws_access_key_id=user_key.id, aws_secret_access_key=user_key.secret)
        try_to_assume_role_without_mfa(role.arn, 'demo-sts-session', sts_client)
        mfa_totp = input('Enter the code from your registered MFA device: ')
        list_buckets_from_assumed_role_with_mfa(role.arn, 'demo-sts-session', virtual_mfa_device.serial_number, mfa_totp, sts_client)
    finally:
        teardown(user, virtual_mfa_device, role)
        print('Thanks for watching!')
if __name__ == '__main__':
    usage_demo()