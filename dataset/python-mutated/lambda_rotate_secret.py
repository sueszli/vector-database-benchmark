import logging
import os
import boto3
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def secret_of_rotation_from_version_id(version_id: str) -> str:
    if False:
        while True:
            i = 10
    return f'lambda_rotate_secret_rotation_{version_id}'

def secret_signal_resource_not_found_exception_on_create(version_id: str) -> str:
    if False:
        return 10
    return f'ResourceNotFoundException_{version_id}'

def handler(event, context):
    if False:
        return 10
    'Secrets Manager Rotation Template\n\n    This is a template for creating an AWS Secrets Manager rotation lambda\n\n    Args:\n        event (dict): Lambda dictionary of event parameters. These keys must include the following:\n            - SecretId: The secret ARN or identifier\n            - ClientRequestToken: The ClientRequestToken of the secret version\n            - Step: The rotation step (one of createSecret, setSecret, testSecret, or finishSecret)\n\n        context (LambdaContext): The Lambda runtime information\n\n    Raises:\n        ResourceNotFoundException: If the secret with the specified arn and stage does not exist\n\n        ValueError: If the secret is not properly configured for rotation\n\n        KeyError: If the event parameters do not contain the expected keys\n\n    '
    endpoint_url = os.environ['AWS_ENDPOINT_URL']
    region = os.environ['AWS_REGION']
    service_client = boto3.client('secretsmanager', endpoint_url=endpoint_url, verify=False, region_name=region)
    arn = event['SecretId']
    token = event['ClientRequestToken']
    step = event['Step']
    metadata = service_client.describe_secret(SecretId=arn)
    if not metadata['RotationEnabled']:
        logger.error(f'Secret {arn} is not enabled for rotation')
        raise ValueError(f'Secret {arn} is not enabled for rotation')
    versions = metadata['VersionIdsToStages']
    if token not in versions:
        logger.error(f'Secret version {token} has no stage for rotation of secret {arn}.')
        raise ValueError(f'Secret version {token} has no stage for rotation of secret {arn}.')
    if 'AWSCURRENT' in versions[token]:
        logger.info(f'Secret version {token} already set as AWSCURRENT for secret {arn}.')
        return
    elif 'AWSPENDING' not in versions[token]:
        logger.error(f'Secret version {token} not set as AWSPENDING for rotation of secret {arn}.')
        raise ValueError(f'Secret version {token} not set as AWSPENDING for rotation of secret {arn}.')
    if step == 'createSecret':
        create_secret(service_client, arn, token)
    elif step == 'setSecret':
        set_secret(service_client, arn, token)
    elif step == 'testSecret':
        test_secret(service_client, arn, token)
    elif step == 'finishSecret':
        finish_secret(service_client, arn, token)
    else:
        raise ValueError('Invalid step parameter')

def create_secret(service_client, arn, token):
    if False:
        while True:
            i = 10
    'Create the secret\n\n    This method first checks for the existence of a secret for the passed in token. If one does not exist, it will generate a\n    new secret and put it with the passed in token.\n\n    Args:\n        service_client (client): The secrets manager service client\n\n        arn (string): The secret ARN or other identifier\n\n        token (string): The ClientRequestToken associated with the secret version\n\n    Raises:\n        ResourceNotFoundException: If the secret with the specified arn and stage does not exist\n\n    '
    service_client.get_secret_value(SecretId=arn, VersionStage='AWSCURRENT')
    try:
        service_client.get_secret_value(SecretId=arn, VersionId=token, VersionStage='AWSPENDING')
        logger.info(f'createSecret: Successfully retrieved secret for {arn}.')
    except service_client.exceptions.ResourceNotFoundException:
        sig_exception = secret_signal_resource_not_found_exception_on_create(token)
        service_client.create_secret(Name=sig_exception, SecretString=sig_exception)
        passwd = secret_of_rotation_from_version_id(token)
        service_client.put_secret_value(SecretId=arn, ClientRequestToken=token, SecretString=passwd, VersionStages=['AWSPENDING'])
        logger.info(f'createSecret: Successfully put secret for ARN {arn} and version {token} with passwd {passwd}.')

def set_secret(service_client, arn, token):
    if False:
        for i in range(10):
            print('nop')
    "Set the secret\n\n    This method should set the AWSPENDING secret in the service that the secret belongs to. For example, if the secret is a database\n    credential, this method should take the value of the AWSPENDING secret and set the user's password to this value in the database.\n\n    Args:\n        service_client (client): The secrets manager service client\n\n        arn (string): The secret ARN or other identifier\n\n        token (string): The ClientRequestToken associated with the secret version\n\n    "
    logger.info('lambda_rotate_secret: set_secret not implemented.')

def test_secret(service_client, arn, token):
    if False:
        while True:
            i = 10
    'Test the secret\n\n    This method should validate that the AWSPENDING secret works in the service that the secret belongs to. For example, if the secret\n    is a database credential, this method should validate that the user can login with the password in AWSPENDING and that the user has\n    all of the expected permissions against the database.\n\n    Args:\n        service_client (client): The secrets manager service client\n\n        arn (string): The secret ARN or other identifier\n\n        token (string): The ClientRequestToken associated with the secret version\n\n    '
    logger.info('lambda_rotate_secret: test_secret not implemented.')

def finish_secret(service_client, arn, token):
    if False:
        return 10
    'Finish the secret\n\n    This method finalizes the rotation process by marking the secret version passed in as the AWSCURRENT secret.\n\n    Args:\n        service_client (client): The secrets manager service client\n\n        arn (string): The secret ARN or other identifier\n\n        token (string): The ClientRequestToken associated with the secret version\n\n    Raises:\n        ResourceNotFoundException: If the secret with the specified arn does not exist\n\n    '
    metadata = service_client.describe_secret(SecretId=arn)
    current_version = None
    for version in metadata['VersionIdsToStages']:
        if 'AWSCURRENT' in metadata['VersionIdsToStages'][version]:
            if version == token:
                logger.info(f'finishSecret: Version {version} already marked as AWSCURRENT for {arn}')
                return
            current_version = version
            break
    service_client.update_secret_version_stage(SecretId=arn, VersionStage='AWSCURRENT', MoveToVersionId=token, RemoveFromVersionId=current_version)
    logger.info(f'finishSecret: Successfully set AWSCURRENT stage to version {token} for secret {arn}.')