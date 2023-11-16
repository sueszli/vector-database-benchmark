from unittest import mock
from boto3 import client
from mock import patch
from moto import mock_cloudtrail, mock_s3
from moto.core import DEFAULT_ACCOUNT_ID
from prowler.providers.aws.services.awslambda.awslambda_service import Function
from tests.providers.aws.audit_info_utils import AWS_REGION_US_EAST_1, set_mocked_aws_audit_info

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        print('Hello World!')
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION_US_EAST_1)
    regional_client.region = AWS_REGION_US_EAST_1
    return {AWS_REGION_US_EAST_1: regional_client}

@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
class Test_awslambda_function_invoke_api_operations_cloudtrail_logging_enabled:

    @mock_cloudtrail
    def test_no_functions(self):
        if False:
            while True:
                i = 10
        lambda_client = mock.MagicMock
        lambda_client.functions = {}
        from prowler.providers.aws.services.cloudtrail.cloudtrail_service import Cloudtrail
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', set_mocked_aws_audit_info()), mock.patch('prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_client', new=lambda_client), mock.patch('prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.cloudtrail_client', new=Cloudtrail(set_mocked_aws_audit_info())):
            from prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled import awslambda_function_invoke_api_operations_cloudtrail_logging_enabled
            check = awslambda_function_invoke_api_operations_cloudtrail_logging_enabled()
            result = check.execute()
            assert len(result) == 0

    @mock_cloudtrail
    @mock_s3
    def test_lambda_not_recorded_by_cloudtrail(self):
        if False:
            i = 10
            return i + 15
        lambda_client = mock.MagicMock
        function_name = 'test-lambda'
        function_runtime = 'python3.9'
        function_arn = f'arn:aws:lambda:{AWS_REGION_US_EAST_1}:{DEFAULT_ACCOUNT_ID}:function/{function_name}'
        lambda_client.functions = {function_name: Function(name=function_name, security_groups=[], arn=function_arn, region=AWS_REGION_US_EAST_1, runtime=function_runtime)}
        cloudtrail_client = client('cloudtrail', region_name=AWS_REGION_US_EAST_1)
        s3_client = client('s3', region_name=AWS_REGION_US_EAST_1)
        trail_name = 'test-trail'
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        cloudtrail_client.create_trail(Name=trail_name, S3BucketName=bucket_name, IsMultiRegionTrail=False)
        from prowler.providers.aws.services.cloudtrail.cloudtrail_service import Cloudtrail
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', set_mocked_aws_audit_info()), mock.patch('prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_client', new=lambda_client), mock.patch('prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.cloudtrail_client', new=Cloudtrail(set_mocked_aws_audit_info())):
            from prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled import awslambda_function_invoke_api_operations_cloudtrail_logging_enabled
            check = awslambda_function_invoke_api_operations_cloudtrail_logging_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].region == AWS_REGION_US_EAST_1
            assert result[0].resource_id == function_name
            assert result[0].resource_arn == function_arn
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'Lambda function {function_name} is not recorded by CloudTrail.'
            assert result[0].resource_tags == []

    @mock_cloudtrail
    @mock_s3
    def test_lambda_recorded_by_cloudtrail_classic_event_selector(self):
        if False:
            for i in range(10):
                print('nop')
        lambda_client = mock.MagicMock
        function_name = 'test-lambda'
        function_runtime = 'python3.9'
        function_arn = f'arn:aws:lambda:{AWS_REGION_US_EAST_1}:{DEFAULT_ACCOUNT_ID}:function/{function_name}'
        lambda_client.functions = {function_name: Function(name=function_name, security_groups=[], arn=function_arn, region=AWS_REGION_US_EAST_1, runtime=function_runtime)}
        cloudtrail_client = client('cloudtrail', region_name=AWS_REGION_US_EAST_1)
        s3_client = client('s3', region_name=AWS_REGION_US_EAST_1)
        trail_name = 'test-trail'
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        cloudtrail_client.create_trail(Name=trail_name, S3BucketName=bucket_name, IsMultiRegionTrail=False)
        _ = cloudtrail_client.put_event_selectors(TrailName=trail_name, EventSelectors=[{'ReadWriteType': 'All', 'IncludeManagementEvents': True, 'DataResources': [{'Type': 'AWS::Lambda::Function', 'Values': [function_arn]}]}])
        from prowler.providers.aws.services.cloudtrail.cloudtrail_service import Cloudtrail
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', set_mocked_aws_audit_info()), mock.patch('prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_client', new=lambda_client), mock.patch('prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.cloudtrail_client', new=Cloudtrail(set_mocked_aws_audit_info())):
            from prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled import awslambda_function_invoke_api_operations_cloudtrail_logging_enabled
            check = awslambda_function_invoke_api_operations_cloudtrail_logging_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].region == AWS_REGION_US_EAST_1
            assert result[0].resource_id == function_name
            assert result[0].resource_arn == function_arn
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'Lambda function {function_name} is recorded by CloudTrail trail {trail_name}.'
            assert result[0].resource_tags == []

    @mock_cloudtrail
    @mock_s3
    def test_lambda_recorded_by_cloudtrail_advanced_event_selector(self):
        if False:
            for i in range(10):
                print('nop')
        lambda_client = mock.MagicMock
        function_name = 'test-lambda'
        function_runtime = 'python3.9'
        function_arn = f'arn:aws:lambda:{AWS_REGION_US_EAST_1}:{DEFAULT_ACCOUNT_ID}:function/{function_name}'
        lambda_client.functions = {function_name: Function(name=function_name, security_groups=[], arn=function_arn, region=AWS_REGION_US_EAST_1, runtime=function_runtime)}
        cloudtrail_client = client('cloudtrail', region_name=AWS_REGION_US_EAST_1)
        s3_client = client('s3', region_name=AWS_REGION_US_EAST_1)
        trail_name = 'test-trail'
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        cloudtrail_client.create_trail(Name=trail_name, S3BucketName=bucket_name, IsMultiRegionTrail=False)
        _ = cloudtrail_client.put_event_selectors(TrailName=trail_name, AdvancedEventSelectors=[{'Name': 'lambda', 'FieldSelectors': [{'Field': 'eventCategory', 'Equals': ['Data']}, {'Field': 'resources.type', 'Equals': ['AWS::Lambda::Function']}]}])
        from prowler.providers.aws.services.cloudtrail.cloudtrail_service import Cloudtrail
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', set_mocked_aws_audit_info()), mock.patch('prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_client', new=lambda_client), mock.patch('prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.cloudtrail_client', new=Cloudtrail(set_mocked_aws_audit_info())):
            from prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled import awslambda_function_invoke_api_operations_cloudtrail_logging_enabled
            check = awslambda_function_invoke_api_operations_cloudtrail_logging_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].region == AWS_REGION_US_EAST_1
            assert result[0].resource_id == function_name
            assert result[0].resource_arn == function_arn
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'Lambda function {function_name} is recorded by CloudTrail trail {trail_name}.'
            assert result[0].resource_tags == []

    @mock_cloudtrail
    @mock_s3
    def test_all_lambdas_recorded_by_cloudtrail(self):
        if False:
            print('Hello World!')
        lambda_client = mock.MagicMock
        function_name = 'test-lambda'
        function_runtime = 'python3.9'
        function_arn = 'arn:aws:lambda'
        lambda_client.functions = {function_name: Function(name=function_name, security_groups=[], arn=function_arn, region=AWS_REGION_US_EAST_1, runtime=function_runtime)}
        cloudtrail_client = client('cloudtrail', region_name=AWS_REGION_US_EAST_1)
        s3_client = client('s3', region_name=AWS_REGION_US_EAST_1)
        trail_name = 'test-trail'
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        cloudtrail_client.create_trail(Name=trail_name, S3BucketName=bucket_name, IsMultiRegionTrail=False)
        _ = cloudtrail_client.put_event_selectors(TrailName=trail_name, EventSelectors=[{'ReadWriteType': 'All', 'IncludeManagementEvents': True, 'DataResources': [{'Type': 'AWS::Lambda::Function', 'Values': [function_arn]}]}])
        from prowler.providers.aws.services.cloudtrail.cloudtrail_service import Cloudtrail
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', set_mocked_aws_audit_info()), mock.patch('prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_client', new=lambda_client), mock.patch('prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.cloudtrail_client', new=Cloudtrail(set_mocked_aws_audit_info())):
            from prowler.providers.aws.services.awslambda.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled.awslambda_function_invoke_api_operations_cloudtrail_logging_enabled import awslambda_function_invoke_api_operations_cloudtrail_logging_enabled
            check = awslambda_function_invoke_api_operations_cloudtrail_logging_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].region == AWS_REGION_US_EAST_1
            assert result[0].resource_id == function_name
            assert result[0].resource_arn == function_arn
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'Lambda function {function_name} is recorded by CloudTrail trail {trail_name}.'
            assert result[0].resource_tags == []