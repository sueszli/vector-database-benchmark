from unittest import skipIf
import pytest
from parameterized import parameterized
from samcli.commands.pipeline.bootstrap.cli import PIPELINE_CONFIG_FILENAME, PIPELINE_CONFIG_DIR
from samcli.lib.config.samconfig import SamConfig
from tests.integration.pipeline.base import BootstrapIntegBase
from tests.testing_utils import run_command_with_input, RUNNING_ON_CI, RUNNING_TEST_FOR_MASTER_ON_CI, RUN_BY_CANARY, run_command, run_command_with_inputs
import boto3
from botocore.exceptions import ClientError
SKIP_BOOTSTRAP_TESTS = RUNNING_ON_CI and RUNNING_TEST_FOR_MASTER_ON_CI and (not RUN_BY_CANARY)
CREDENTIAL_PROFILE = '2' if not RUN_BY_CANARY else '1'
CFN_OUTPUT_TO_CONFIG_KEY = {'ArtifactsBucket': 'artifacts_bucket', 'CloudFormationExecutionRole': 'cloudformation_execution_role', 'PipelineExecutionRole': 'pipeline_execution_role', 'PipelineUser': 'pipeline_user'}

@skipIf(SKIP_BOOTSTRAP_TESTS, 'Skip bootstrap tests in CI/CD only')
class TestBootstrap(BootstrapIntegBase):

    @parameterized.expand([('create_image_repository',), (False,)])
    def test_interactive_with_no_resources_provided(self, create_image_repository):
        if False:
            return 10
        (stage_configuration_name, stack_name) = self._get_stage_and_stack_name()
        self.stack_names = [stack_name]
        bootstrap_command_list = self.get_bootstrap_command_list()
        inputs = [stage_configuration_name, CREDENTIAL_PROFILE, self.region, '1', '', '', '', '', 'y' if create_image_repository else 'N']
        if create_image_repository:
            inputs.append('')
        inputs.append('')
        inputs.append('y')
        bootstrap_process_execute = run_command_with_inputs(bootstrap_command_list, inputs)
        self.assertEqual(bootstrap_process_execute.process.returncode, 0)
        stdout = bootstrap_process_execute.stdout.decode()
        self.assertIn('ACCESS_KEY_ID', stdout)
        self.assertIn('SECRET_ACCESS_KEY', stdout)
        common_resources = {'PipelineUser', 'PipelineUserAccessKey', 'PipelineUserSecretKey', 'CloudFormationExecutionRole', 'PipelineExecutionRole', 'ArtifactsBucket', 'ArtifactsLoggingBucket', 'ArtifactsLoggingBucketPolicy', 'ArtifactsBucketPolicy', 'PipelineExecutionRolePermissionPolicy'}
        if create_image_repository:
            self.assertSetEqual({*common_resources, 'ImageRepository'}, set(self._extract_created_resource_logical_ids(stack_name)))
            CFN_OUTPUT_TO_CONFIG_KEY['ImageRepository'] = 'image_repository'
            self.validate_pipeline_config(stack_name, stage_configuration_name, list(CFN_OUTPUT_TO_CONFIG_KEY.keys()))
            del CFN_OUTPUT_TO_CONFIG_KEY['ImageRepository']
        else:
            self.assertSetEqual(common_resources, set(self._extract_created_resource_logical_ids(stack_name)))
            self.validate_pipeline_config(stack_name, stage_configuration_name)

    def test_interactive_with_no_resources_provided_using_oidc(self):
        if False:
            for i in range(10):
                print('nop')
        (stage_configuration_name, stack_name) = self._get_stage_and_stack_name()
        self.stack_names = [stack_name]
        bootstrap_command_list = self.get_bootstrap_command_list()
        inputs = [stage_configuration_name, CREDENTIAL_PROFILE, self.region, '2', '1', 'https://token.actions.githubusercontent.com', 'sts.amazonaws.com', 'test_github_org', 'test_not_real', 'main', '', '', '', 'N']
        inputs.append('')
        inputs.append('y')
        bootstrap_process_execute = run_command_with_inputs(bootstrap_command_list, inputs)
        self.assertEqual(bootstrap_process_execute.process.returncode, 0)
        stdout = bootstrap_process_execute.stdout.decode()
        common_resources = {'CloudFormationExecutionRole', 'PipelineExecutionRole', 'ArtifactsBucket', 'ArtifactsLoggingBucket', 'ArtifactsLoggingBucketPolicy', 'ArtifactsBucketPolicy', 'PipelineExecutionRolePermissionPolicy', 'OidcProvider'}
        CFN_OUTPUT_TO_CONFIG_KEY['OidcProvider'] = 'oidc_provider_url'
        del CFN_OUTPUT_TO_CONFIG_KEY['PipelineUser']
        self.assertSetEqual(common_resources, set(self._extract_created_resource_logical_ids(stack_name)))
        self.validate_pipeline_config(stack_name, stage_configuration_name)
        del CFN_OUTPUT_TO_CONFIG_KEY['OidcProvider']
        CFN_OUTPUT_TO_CONFIG_KEY['PipelineUser'] = 'pipeline_user'

    @parameterized.expand([('create_image_repository',), (False,)])
    def test_non_interactive_with_no_resources_provided(self, create_image_repository):
        if False:
            return 10
        (stage_configuration_name, stack_name) = self._get_stage_and_stack_name()
        self.stack_names = [stack_name]
        bootstrap_command_list = self.get_bootstrap_command_list(no_interactive=True, create_image_repository=create_image_repository, no_confirm_changeset=True, region=self.region)
        bootstrap_process_execute = run_command(bootstrap_command_list)
        self.assertEqual(bootstrap_process_execute.process.returncode, 2)
        stderr = bootstrap_process_execute.stderr.decode()
        self.assertIn('Missing required parameter', stderr)

    def test_interactive_with_all_required_resources_provided(self):
        if False:
            return 10
        (stage_configuration_name, stack_name) = self._get_stage_and_stack_name()
        self.stack_names = [stack_name]
        bootstrap_command_list = self.get_bootstrap_command_list()
        inputs = [stage_configuration_name, CREDENTIAL_PROFILE, self.region, '1', 'arn:aws:iam::123:user/user-name', 'arn:aws:iam::123:role/role-name', 'arn:aws:iam::123:role/role-name', 'arn:aws:s3:::bucket-name', 'N', '']
        bootstrap_process_execute = run_command_with_inputs(bootstrap_command_list, inputs)
        self.assertEqual(bootstrap_process_execute.process.returncode, 0)
        stdout = bootstrap_process_execute.stdout.decode()
        self.assertIn('skipping creation', stdout)

    def test_no_interactive_with_all_required_resources_provided(self):
        if False:
            i = 10
            return i + 15
        (stage_configuration_name, stack_name) = self._get_stage_and_stack_name()
        self.stack_names = [stack_name]
        bootstrap_command_list = self.get_bootstrap_command_list(no_interactive=True, stage_configuration_name=stage_configuration_name, pipeline_user='arn:aws:iam::123:user/user-name', pipeline_execution_role='arn:aws:iam::123:role/role-name', cloudformation_execution_role='arn:aws:iam::123:role/role-name', bucket='arn:aws:s3:::bucket-name', image_repository='arn:aws:ecr:::repository/repo-name', region=self.region)
        bootstrap_process_execute = run_command(bootstrap_command_list)
        self.assertEqual(bootstrap_process_execute.process.returncode, 0)
        stdout = bootstrap_process_execute.stdout.decode()
        self.assertIn('skipping creation', stdout)

    def validate_pipeline_config(self, stack_name, stage_configuration_name, cfn_keys_to_check=None):
        if False:
            i = 10
            return i + 15
        if cfn_keys_to_check is None:
            cfn_keys_to_check = list(CFN_OUTPUT_TO_CONFIG_KEY.keys())
        response = self.cf_client.describe_stacks(StackName=stack_name)
        stacks = response['Stacks']
        self.assertTrue(len(stacks) > 0)
        stack_outputs = stacks[0]['Outputs']
        output_values = {}
        for value in stack_outputs:
            output_values[value['OutputKey']] = value['OutputValue']
        config = SamConfig(PIPELINE_CONFIG_DIR, PIPELINE_CONFIG_FILENAME)
        config_values = config.get_all(['pipeline', 'bootstrap'], 'parameters', stage_configuration_name)
        config_values = {**config_values, **config.get_all(['pipeline', 'bootstrap'], 'parameters')}
        for key in CFN_OUTPUT_TO_CONFIG_KEY:
            if key not in cfn_keys_to_check:
                continue
            value = CFN_OUTPUT_TO_CONFIG_KEY[key]
            if key != 'OidcProvider':
                cfn_value = output_values[key]
            config_value = config_values[value]
            if key == 'ImageRepository':
                self.assertEqual(cfn_value.split('/')[-1], config_value.split('/')[-1])
            elif key == 'OidcProvider':
                self.assertTrue(config_value.startswith('https://'))
            else:
                self.assertTrue(cfn_value.endswith(config_value) or cfn_value == config_value)

    @parameterized.expand([('confirm_changeset',), (False,)])
    def test_no_interactive_with_some_required_resources_provided(self, confirm_changeset: bool):
        if False:
            while True:
                i = 10
        (stage_configuration_name, stack_name) = self._get_stage_and_stack_name()
        self.stack_names = [stack_name]
        bootstrap_command_list = self.get_bootstrap_command_list(no_interactive=True, stage_configuration_name=stage_configuration_name, pipeline_user='arn:aws:iam::123:user/user-name', pipeline_execution_role='arn:aws:iam::123:role/role-name', bucket='arn:aws:s3:::bucket-name', image_repository='arn:aws:ecr:::repository/repo-name', no_confirm_changeset=not confirm_changeset, region=self.region)
        inputs = ['y']
        bootstrap_process_execute = run_command_with_inputs(bootstrap_command_list, inputs if confirm_changeset else [])
        self.assertEqual(bootstrap_process_execute.process.returncode, 0)
        stdout = bootstrap_process_execute.stdout.decode()
        self.assertIn('Successfully created!', stdout)
        self.assertIn('CloudFormationExecutionRole', self._extract_created_resource_logical_ids(stack_name))

    def test_interactive_cancelled_by_user(self):
        if False:
            while True:
                i = 10
        (stage_configuration_name, stack_name) = self._get_stage_and_stack_name()
        self.stack_names = [stack_name]
        bootstrap_command_list = self.get_bootstrap_command_list()
        inputs = [stage_configuration_name, CREDENTIAL_PROFILE, self.region, '1', 'arn:aws:iam::123:user/user-name', 'arn:aws:iam::123:role/role-name', '', 'arn:aws:s3:::bucket-name', 'N', '', '']
        bootstrap_process_execute = run_command_with_inputs(bootstrap_command_list, inputs)
        self.assertEqual(bootstrap_process_execute.process.returncode, 0)
        stdout = bootstrap_process_execute.stdout.decode()
        self.assertTrue(stdout.strip().endswith('Canceling pipeline bootstrap creation.'))
        self.assertFalse(self._stack_exists(stack_name))

    def test_interactive_with_some_required_resources_provided(self):
        if False:
            return 10
        (stage_configuration_name, stack_name) = self._get_stage_and_stack_name()
        self.stack_names = [stack_name]
        bootstrap_command_list = self.get_bootstrap_command_list()
        inputs = [stage_configuration_name, CREDENTIAL_PROFILE, self.region, '1', 'arn:aws:iam::123:user/user-name', 'arn:aws:iam::123:role/role-name', '', 'arn:aws:s3:::bucket-name', 'N', '', 'y']
        bootstrap_process_execute = run_command_with_inputs(bootstrap_command_list, inputs)
        self.assertEqual(bootstrap_process_execute.process.returncode, 0)
        stdout = bootstrap_process_execute.stdout.decode()
        self.assertIn('Successfully created!', stdout)
        self.assertIn('CloudFormationExecutionRole', self._extract_created_resource_logical_ids(stack_name))
        if 'ImageRepository' in CFN_OUTPUT_TO_CONFIG_KEY:
            del CFN_OUTPUT_TO_CONFIG_KEY['ImageRepository']
        if 'OidcProvider' in CFN_OUTPUT_TO_CONFIG_KEY:
            del CFN_OUTPUT_TO_CONFIG_KEY['OidcProvider']
        self.validate_pipeline_config(stack_name, stage_configuration_name)

    def test_interactive_pipeline_user_only_created_once(self):
        if False:
            print('Hello World!')
        '\n        Create 3 stages, only the first stage resource stack creates\n        a pipeline user, and the remaining two share the same pipeline user.\n        '
        stage_configuration_names = []
        for suffix in ['1', '2', '3']:
            (stage_configuration_name, stack_name) = self._get_stage_and_stack_name(suffix)
            stage_configuration_names.append(stage_configuration_name)
            self.stack_names.append(stack_name)
        bootstrap_command_list = self.get_bootstrap_command_list()
        for (i, stage_configuration_name) in enumerate(stage_configuration_names):
            inputs = [stage_configuration_name, CREDENTIAL_PROFILE, self.region, '1', *([''] if i == 0 else []), 'arn:aws:iam::123:role/role-name', 'arn:aws:iam::123:role/role-name', 'arn:aws:s3:::bucket-name', 'N', '', 'y']
            bootstrap_process_execute = run_command_with_input(bootstrap_command_list, ('\n'.join(inputs) + '\n').encode())
            self.assertEqual(bootstrap_process_execute.process.returncode, 0)
            stdout = bootstrap_process_execute.stdout.decode()
            if i == 0:
                self.assertIn('The following resources were created in your account:', stdout)
                resources = self._extract_created_resource_logical_ids(self.stack_names[i])
                self.assertTrue('PipelineUser' in resources)
                self.assertTrue('PipelineUserAccessKey' in resources)
                self.assertTrue('PipelineUserSecretKey' in resources)
                self.validate_pipeline_config(self.stack_names[i], stage_configuration_name)
            else:
                self.assertIn('skipping creation', stdout)

    @parameterized.expand([('ArtifactsBucket',), ('ArtifactsLoggingBucket',)])
    def test_bootstrapped_buckets_accept_ssl_requests_only(self, bucket_logical_id):
        if False:
            print('Hello World!')
        (stage_configuration_name, stack_name) = self._get_stage_and_stack_name()
        self.stack_names = [stack_name]
        bootstrap_command_list = self.get_bootstrap_command_list(stage_configuration_name=stage_configuration_name, no_interactive=True, no_confirm_changeset=True, region=self.region)
        bootstrap_process_execute = run_command(bootstrap_command_list)
        self.assertEqual(bootstrap_process_execute.process.returncode, 0)
        stack_resources = self.cf_client.describe_stack_resources(StackName=stack_name)
        bucket = next((resource for resource in stack_resources['StackResources'] if resource['LogicalResourceId'] == bucket_logical_id))
        bucket_name = bucket['PhysicalResourceId']
        bucket_key = 'any/testing/key.txt'
        testing_data = b'any testing binary data'
        s3_ssl_client = boto3.client('s3', region_name=self.region)
        s3_non_ssl_client = boto3.client('s3', use_ssl=False, region_name=self.region)
        s3_ssl_client.put_object(Body=testing_data, Bucket=bucket_name, Key=bucket_key)
        res = s3_ssl_client.get_object(Bucket=bucket_name, Key=bucket_key)
        retrieved_data = res['Body'].read()
        self.assertEqual(retrieved_data, testing_data)
        with self.assertRaises(ClientError) as error:
            s3_non_ssl_client.get_object(Bucket=bucket_name, Key=bucket_key)
        self.assertEqual(str(error.exception), 'An error occurred (AccessDenied) when calling the GetObject operation: Access Denied')

    def test_bootstrapped_artifacts_bucket_has_server_access_log_enabled(self):
        if False:
            print('Hello World!')
        (stage_configuration_name, stack_name) = self._get_stage_and_stack_name()
        self.stack_names = [stack_name]
        bootstrap_command_list = self.get_bootstrap_command_list(stage_configuration_name=stage_configuration_name, no_interactive=True, no_confirm_changeset=True, region=self.region)
        bootstrap_process_execute = run_command(bootstrap_command_list)
        self.assertEqual(bootstrap_process_execute.process.returncode, 0)
        stack_resources = self.cf_client.describe_stack_resources(StackName=stack_name)
        artifacts_bucket = next((resource for resource in stack_resources['StackResources'] if resource['LogicalResourceId'] == 'ArtifactsBucket'))
        artifacts_bucket_name = artifacts_bucket['PhysicalResourceId']
        artifacts_logging_bucket = next((resource for resource in stack_resources['StackResources'] if resource['LogicalResourceId'] == 'ArtifactsLoggingBucket'))
        artifacts_logging_bucket_name = artifacts_logging_bucket['PhysicalResourceId']
        s3_client = boto3.client('s3', region_name=self.region)
        res = s3_client.get_bucket_logging(Bucket=artifacts_bucket_name)
        self.assertEqual(artifacts_logging_bucket_name, res['LoggingEnabled']['TargetBucket'])