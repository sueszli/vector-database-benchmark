import os
import uuid
import json
import time
from pathlib import Path
from unittest import TestCase
import boto3
from botocore.exceptions import ClientError
from samcli.lib.bootstrap.companion_stack.data_types import CompanionStack
from tests.testing_utils import method_to_stack_name, get_sam_command
SLEEP = 3

class PackageIntegBase(TestCase):
    kms_key = None
    ecr_repo_name = None

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.region_name = os.environ.get('AWS_DEFAULT_REGION')
        '\n        Our integration tests use S3 bucket and ECR Repo to run several tests.\n        Given that S3 objects are eventually consistent and we are using same bucket for\n        lot of integration tests, we want to have multiple buckets to reduce\n        transient failures. In order to achieve this we created 3 buckets one for each python version we support (3.7,\n        3.8 and 3.9). Tests running for respective python version will use respective bucket.\n\n        AWS_S3 will point to a new environment variable AWS_S3_36 or AWS_S3_37 or AWS_S3_38. This is controlled by\n        Appveyor. These environment variables will hold bucket name to run integration tests. Eg:\n\n        For Python36:\n        AWS_S3=AWS_S3_36\n        AWS_S3_36=aws-sam-cli-canary-region-awssamclitestbucket-forpython36\n\n        AWS_ECR will point to a new environment variable AWS_ECR_36 or AWS_ECR_37 or AWS_ECR_38. This is controlled by\n        Appveyor. These environment variables will hold bucket name to run integration tests. Eg:\n\n        For Python36:\n        AWS_S3=AWS_ECR_36\n        AWS_S3_36=123456789012.dkr.ecr.us-east-1.amazonaws.com/sam-cli-py36\n\n        For backwards compatibility we are falling back to reading AWS_S3 so that current tests keep working.\n        For backwards compatibility we are falling back to reading AWS_ECR so that current tests keep working.\n        '
        s3_bucket_from_env_var = os.environ.get('AWS_S3')
        ecr_repo_from_env_var = os.environ.get('AWS_ECR')
        if s3_bucket_from_env_var:
            cls.pre_created_bucket = os.environ.get(s3_bucket_from_env_var, False)
        else:
            cls.pre_created_bucket = False
        if ecr_repo_from_env_var:
            cls.pre_created_ecr_repo = os.environ.get(ecr_repo_from_env_var, False)
        else:
            cls.pre_created_ecr_repo = False
        cls.ecr_repo_name = cls.pre_created_ecr_repo if cls.pre_created_ecr_repo else str(uuid.uuid4()).replace('-', '')[:10]
        cls.bucket_name = cls.pre_created_bucket if cls.pre_created_bucket else str(uuid.uuid4())
        cls.test_data_path = Path(__file__).resolve().parents[1].joinpath('testdata', 'package')
        s3 = boto3.resource('s3')
        cls.ecr = boto3.client('ecr')
        cls.kms_key = os.environ.get('AWS_KMS_KEY')
        cls.s3_bucket = s3.Bucket(cls.bucket_name)
        if not cls.pre_created_bucket:
            cls.s3_bucket.create()
            time.sleep(SLEEP)
            bucket_versioning = s3.BucketVersioning(cls.bucket_name)
            bucket_versioning.enable()
            time.sleep(SLEEP)
        if not cls.pre_created_ecr_repo:
            ecr_result = cls.ecr.create_repository(repositoryName=cls.ecr_repo_name)
            cls.ecr_repo_name = ecr_result.get('repository', {}).get('repositoryUri', None)
            time.sleep(SLEEP)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.s3_prefix = uuid.uuid4().hex
        super().setUp()

    @staticmethod
    def get_command_list(s3_bucket=None, template=None, template_file=None, s3_prefix=None, output_template_file=None, use_json=False, force_upload=False, no_progressbar=False, kms_key_id=None, metadata=None, image_repository=None, image_repositories=None, resolve_s3=False):
        if False:
            while True:
                i = 10
        command_list = [get_sam_command(), 'package']
        if s3_bucket:
            command_list = command_list + ['--s3-bucket', str(s3_bucket)]
        if template:
            command_list = command_list + ['--template', str(template)]
        if template_file:
            command_list = command_list + ['--template-file', str(template_file)]
        if s3_prefix:
            command_list = command_list + ['--s3-prefix', str(s3_prefix)]
        if output_template_file:
            command_list = command_list + ['--output-template-file', str(output_template_file)]
        if kms_key_id:
            command_list = command_list + ['--kms-key-id', str(kms_key_id)]
        if use_json:
            command_list = command_list + ['--use-json']
        if force_upload:
            command_list = command_list + ['--force-upload']
        if no_progressbar:
            command_list = command_list + ['--no-progressbar']
        if metadata:
            command_list = command_list + ['--metadata', json.dumps(metadata)]
        if image_repository:
            command_list = command_list + ['--image-repository', str(image_repository)]
        if image_repositories:
            command_list = command_list + ['--image-repositories', str(image_repositories)]
        if resolve_s3:
            command_list = command_list + ['--resolve-s3']
        return command_list

    def _method_to_stack_name(self, method_name):
        if False:
            print('Hello World!')
        return method_to_stack_name(method_name)

    def _stack_name_to_companion_stack(self, stack_name):
        if False:
            for i in range(10):
                print('nop')
        return CompanionStack(stack_name).stack_name

    def _delete_companion_stack(self, cfn_client, ecr_client, companion_stack_name):
        if False:
            return 10
        repos = list()
        try:
            cfn_client.describe_stacks(StackName=companion_stack_name)
        except ClientError:
            return
        stack = boto3.resource('cloudformation').Stack(companion_stack_name)
        resources = stack.resource_summaries.all()
        for resource in resources:
            if resource.resource_type == 'AWS::ECR::Repository':
                repos.append(resource.physical_resource_id)
        for repo in repos:
            try:
                ecr_client.delete_repository(repositoryName=repo, force=True)
            except ecr_client.exceptions.RepositoryNotFoundException:
                pass
        cfn_client.delete_stack(StackName=companion_stack_name)

    def _assert_companion_stack(self, cfn_client, companion_stack_name):
        if False:
            return 10
        try:
            cfn_client.describe_stacks(StackName=companion_stack_name)
        except ClientError:
            self.fail('No companion stack found.')

    def _assert_companion_stack_content(self, ecr_client, companion_stack_name):
        if False:
            i = 10
            return i + 15
        stack = boto3.resource('cloudformation').Stack(companion_stack_name)
        resources = stack.resource_summaries.all()
        for resource in resources:
            if resource.resource_type == 'AWS::ECR::Repository':
                policy = ecr_client.get_repository_policy(repositoryName=resource.physical_resource_id)
                self._assert_ecr_lambda_policy(policy)
            else:
                self.fail('Non ECR Repo resource found in companion stack')

    def _assert_ecr_lambda_policy(self, policy):
        if False:
            i = 10
            return i + 15
        policyText = json.loads(policy.get('policyText', '{}'))
        statements = policyText.get('Statement')
        self.assertEqual(len(statements), 1)
        lambda_policy = statements[0]
        self.assertEqual(lambda_policy.get('Principal'), {'Service': 'lambda.amazonaws.com'})
        actions = lambda_policy.get('Action')
        self.assertEqual(sorted(actions), sorted(['ecr:GetDownloadUrlForLayer', 'ecr:GetRepositoryPolicy', 'ecr:BatchGetImage']))