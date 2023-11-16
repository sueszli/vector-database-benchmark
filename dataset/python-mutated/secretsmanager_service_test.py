import io
import zipfile
from unittest.mock import patch
from boto3 import client, resource, session
from moto import mock_ec2, mock_iam, mock_lambda, mock_s3, mock_secretsmanager
from moto.core import DEFAULT_ACCOUNT_ID
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.secretsmanager.secretsmanager_service import SecretsManager
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'eu-west-1'

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        for i in range(10):
            print('nop')
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
class Test_SecretsManager_Service:

    def set_mocked_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=DEFAULT_ACCOUNT_ID, audited_account_arn=f'arn:aws:iam::{DEFAULT_ACCOUNT_ID}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_secretsmanager
    def test__get_client__(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        secretsmanager = SecretsManager(audit_info)
        assert secretsmanager.regional_clients[AWS_REGION].__class__.__name__ == 'SecretsManager'

    @mock_secretsmanager
    def test__get_session__(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        secretsmanager = SecretsManager(audit_info)
        assert secretsmanager.session.__class__.__name__ == 'Session'

    @mock_secretsmanager
    def test__get_service__(self):
        if False:
            print('Hello World!')
        audit_info = self.set_mocked_audit_info()
        secretsmanager = SecretsManager(audit_info)
        assert secretsmanager.service == 'secretsmanager'

    @mock_secretsmanager
    @mock_lambda
    @mock_ec2
    @mock_iam
    @mock_s3
    def test__list_secrets__(self):
        if False:
            for i in range(10):
                print('nop')
        secretsmanager_client = client('secretsmanager', region_name=AWS_REGION)
        resp = secretsmanager_client.create_secret(Name='test-secret', SecretString='test-secret', Tags=[{'Key': 'test', 'Value': 'test'}])
        secret_arn = resp['ARN']
        secret_name = resp['Name']
        iam_client = client('iam', region_name=AWS_REGION)
        iam_role = iam_client.create_role(RoleName='rotation-lambda-role', AssumeRolePolicyDocument='test-policy', Path='/')['Role']['Arn']
        s3_client = resource('s3', region_name=AWS_REGION)
        s3_client.create_bucket(Bucket='test-bucket', CreateBucketConfiguration={'LocationConstraint': AWS_REGION})
        zip_output = io.BytesIO()
        zip_file = zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED)
        zip_file.writestr('lambda_function.py', '\n            def lambda_handler(event, context):\n                print("custom log event")\n                return event\n            ')
        zip_file.close()
        zip_output.seek(0)
        lambda_client = client('lambda', region_name=AWS_REGION)
        resp = lambda_client.create_function(FunctionName='rotation-lambda', Runtime='python3.7', Role=iam_role, Handler='lambda_function.lambda_handler', Code={'ZipFile': zip_output.read()}, Description='test lambda function', Timeout=3, MemorySize=128, PackageType='ZIP', Publish=True, VpcConfig={'SecurityGroupIds': ['sg-123abc'], 'SubnetIds': ['subnet-123abc']})
        lambda_arn = resp['FunctionArn']
        secretsmanager_client.rotate_secret(SecretId=secret_arn, RotationLambdaARN=lambda_arn, RotationRules={'AutomaticallyAfterDays': 90, 'Duration': '3h', 'ScheduleExpression': 'rate(10 days)'}, RotateImmediately=True)
        audit_info = self.set_mocked_audit_info()
        secretsmanager = SecretsManager(audit_info)
        assert len(secretsmanager.secrets) == 1
        assert secretsmanager.secrets
        assert secretsmanager.secrets[secret_arn]
        assert secretsmanager.secrets[secret_arn].name == secret_name
        assert secretsmanager.secrets[secret_arn].arn == secret_arn
        assert secretsmanager.secrets[secret_arn].region == AWS_REGION
        assert secretsmanager.secrets[secret_arn].rotation_enabled is True
        assert secretsmanager.secrets[secret_arn].tags == [{'Key': 'test', 'Value': 'test'}]