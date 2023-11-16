"""
Purpose

Shows how to construct a URL that gives federated users direct access to the
AWS Management Console.
"""
import datetime
import json
import sys
import time
import urllib.parse
import boto3
import requests

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
        i = 10
        return i + 15
    return f'demo-assume-role-{base_name}-{time.time_ns()}'

def setup(iam_resource):
    if False:
        while True:
            i = 10
    '\n    Creates a role that can be assumed by the current user.\n    Attaches a policy that allows only Amazon S3 read-only access.\n\n    :param iam_resource: A Boto3 AWS Identity and Access Management (IAM) instance\n                         that has the permission to create a role.\n    :return: The newly created role.\n    '
    role = iam_resource.create_role(RoleName=unique_name('role'), AssumeRolePolicyDocument=json.dumps({'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'AWS': iam_resource.CurrentUser().arn}, 'Action': 'sts:AssumeRole'}]}))
    role.attach_policy(PolicyArn='arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess')
    print(f'Created role {role.name}.')
    print('Give AWS time to propagate these new resources and connections.', end='')
    progress_bar(10)
    return role

def construct_federated_url(assume_role_arn, session_name, issuer, sts_client):
    if False:
        print('Hello World!')
    '\n    Constructs a URL that gives federated users direct access to the AWS Management\n    Console.\n\n    1. Acquires temporary credentials from AWS Security Token Service (AWS STS) that\n       can be used to assume a role with limited permissions.\n    2. Uses the temporary credentials to request a sign-in token from the\n       AWS federation endpoint.\n    3. Builds a URL that can be used in a browser to navigate to the AWS federation\n       endpoint, includes the sign-in token for authentication, and redirects to\n       the AWS Management Console with permissions defined by the role that was\n       specified in step 1.\n\n    :param assume_role_arn: The role that specifies the permissions that are granted.\n                            The current user must have permission to assume the role.\n    :param session_name: The name for the STS session.\n    :param issuer: The organization that issues the URL.\n    :param sts_client: A Boto3 STS instance that can assume the role.\n    :return: The federated URL.\n    '
    response = sts_client.assume_role(RoleArn=assume_role_arn, RoleSessionName=session_name)
    temp_credentials = response['Credentials']
    print(f'Assumed role {assume_role_arn} and got temporary credentials.')
    session_data = {'sessionId': temp_credentials['AccessKeyId'], 'sessionKey': temp_credentials['SecretAccessKey'], 'sessionToken': temp_credentials['SessionToken']}
    aws_federated_signin_endpoint = 'https://signin.aws.amazon.com/federation'
    response = requests.get(aws_federated_signin_endpoint, params={'Action': 'getSigninToken', 'SessionDuration': str(datetime.timedelta(hours=12).seconds), 'Session': json.dumps(session_data)})
    signin_token = json.loads(response.text)
    print(f'Got a sign-in token from the AWS sign-in federation endpoint.')
    query_string = urllib.parse.urlencode({'Action': 'login', 'Issuer': issuer, 'Destination': 'https://console.aws.amazon.com/', 'SigninToken': signin_token['SigninToken']})
    federated_url = f'{aws_federated_signin_endpoint}?{query_string}'
    return federated_url

def teardown(role):
    if False:
        print('Hello World!')
    '\n    Removes all resources created during setup.\n\n    :param role: The demo role.\n    '
    for attached in role.attached_policies.all():
        role.detach_policy(PolicyArn=attached.arn)
        print(f'Detached {attached.policy_name}.')
    role.delete()
    print(f'Deleted {role.name}.')

def usage_demo():
    if False:
        i = 10
        return i + 15
    'Drives the demonstration.'
    print('-' * 88)
    print(f'Welcome to the AWS Security Token Service federated URL demo.')
    print('-' * 88)
    iam_resource = boto3.resource('iam')
    role = setup(iam_resource)
    sts_client = boto3.client('sts')
    try:
        federated_url = construct_federated_url(role.arn, 'AssumeRoleDemoSession', 'example.org', sts_client)
        print('Constructed a federated URL that can be used to connect to the AWS Management Console with role-defined permissions:')
        print('-' * 88)
        print(federated_url)
        print('-' * 88)
        _ = input('Copy and paste the above URL into a browser to open the AWS Management Console with limited permissions. When done, press Enter to clean up and complete this demo.')
    finally:
        teardown(role)
        print('Thanks for watching!')
if __name__ == '__main__':
    usage_demo()