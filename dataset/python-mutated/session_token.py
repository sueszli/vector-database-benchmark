"""
Purpose

Shows how to get a session token that requires a multi-factor authentication (MFA)
token, using AWS Security Token Service (AWS STS) credentials.
"""
import json
import os
import sys
import time
import webbrowser
import boto3
from botocore.exceptions import ClientError

def progress_bar(seconds):
    if False:
        print('Hello World!')
    'Shows a simple progress bar in the command window.'
    for _ in range(seconds):
        time.sleep(1)
        print('.', end='')
        sys.stdout.flush()
    print()

def unique_name(base_name):
    if False:
        return 10
    return f'demo-assume-role-{base_name}-{time.time_ns()}'

def setup(iam_resource):
    if False:
        while True:
            i = 10
    '\n    Creates a new user with no permissions.\n    Creates a new virtual multi-factor authentication (MFA) device.\n    Displays the QR code to seed the device.\n    Asks for two codes from the MFA device.\n    Registers the MFA device for the user.\n    Creates an access key pair for the user.\n    Creates an inline policy for the user that lets the user list Amazon S3 buckets,\n    but only when MFA credentials are used.\n\n    Any MFA device that can scan a QR code will work with this demonstration.\n    Common choices are mobile apps like LastPass Authenticator,\n    Microsoft Authenticator, or Google Authenticator.\n\n    :param iam_resource: A Boto3 AWS Identity and Access Management (IAM) resource\n                         that has permissions to create users, MFA devices, and\n                         policies in the account.\n    :return: The newly created user, user key, and virtual MFA device.\n    '
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
    user.create_policy(PolicyName=unique_name('user-policy'), PolicyDocument=json.dumps({'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 's3:ListAllMyBuckets', 'Resource': 'arn:aws:s3:::*', 'Condition': {'Bool': {'aws:MultiFactorAuthPresent': True}}}]}))
    print(f'Created an inline policy for {user.name} that lets the user list buckets, but only when MFA credentials are present.')
    print('Give AWS time to propagate these new resources and connections.', end='')
    progress_bar(10)
    return (user, user_key, virtual_mfa_device)

def list_buckets_with_session_token_with_mfa(mfa_serial_number, mfa_totp, sts_client):
    if False:
        for i in range(10):
            print('nop')
    '\n    Gets a session token with MFA credentials and uses the temporary session\n    credentials to list Amazon S3 buckets.\n\n    Requires an MFA device serial number and token.\n\n    :param mfa_serial_number: The serial number of the MFA device. For a virtual MFA\n                              device, this is an Amazon Resource Name (ARN).\n    :param mfa_totp: A time-based, one-time password issued by the MFA device.\n    :param sts_client: A Boto3 STS instance that has permission to assume the role.\n    '
    if mfa_serial_number is not None:
        response = sts_client.get_session_token(SerialNumber=mfa_serial_number, TokenCode=mfa_totp)
    else:
        response = sts_client.get_session_token()
    temp_credentials = response['Credentials']
    s3_resource = boto3.resource('s3', aws_access_key_id=temp_credentials['AccessKeyId'], aws_secret_access_key=temp_credentials['SecretAccessKey'], aws_session_token=temp_credentials['SessionToken'])
    print(f'Buckets for the account:')
    for bucket in s3_resource.buckets.all():
        print(bucket.name)

def teardown(user, virtual_mfa_device):
    if False:
        i = 10
        return i + 15
    '\n    Removes all resources created during setup.\n\n    :param user: The demo user.\n    :param role: The demo MFA device.\n    '
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
        print('Hello World!')
    'Drives the demonstration.'
    print('-' * 88)
    print(f'Welcome to the AWS Security Token Service assume role demo, starring multi-factor authentication (MFA)!')
    print('-' * 88)
    iam_resource = boto3.resource('iam')
    (user, user_key, virtual_mfa_device) = setup(iam_resource)
    try:
        sts_client = boto3.client('sts', aws_access_key_id=user_key.id, aws_secret_access_key=user_key.secret)
        try:
            print('Listing buckets without specifying MFA credentials.')
            list_buckets_with_session_token_with_mfa(None, None, sts_client)
        except ClientError as error:
            if error.response['Error']['Code'] == 'AccessDenied':
                print('Got expected AccessDenied error.')
        mfa_totp = input('Enter the code from your registered MFA device: ')
        list_buckets_with_session_token_with_mfa(virtual_mfa_device.serial_number, mfa_totp, sts_client)
    finally:
        teardown(user, virtual_mfa_device)
        print('Thanks for watching!')
if __name__ == '__main__':
    usage_demo()