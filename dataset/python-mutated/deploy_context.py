"""
Deploy a SAM stack
"""
import logging
import os
from typing import Dict, List, Optional
import boto3
import click
from samcli.commands.deploy import exceptions as deploy_exceptions
from samcli.commands.deploy.auth_utils import auth_per_resource
from samcli.commands.deploy.utils import hide_noecho_parameter_overrides, print_deploy_args, sanitize_parameter_overrides
from samcli.lib.deploy.deployer import Deployer
from samcli.lib.deploy.utils import FailureMode
from samcli.lib.intrinsic_resolver.intrinsics_symbol_table import IntrinsicsSymbolTable
from samcli.lib.package.s3_uploader import S3Uploader
from samcli.lib.providers.sam_stack_provider import SamLocalStackProvider
from samcli.lib.utils.boto_utils import get_boto_config_with_user_agent
from samcli.yamlhelper import yaml_parse
LOG = logging.getLogger(__name__)

class DeployContext:
    MSG_SHOWCASE_CHANGESET = '\nChangeset created successfully. {changeset_id}\n'
    MSG_EXECUTE_SUCCESS = '\nSuccessfully created/updated stack - {stack_name} in {region}\n'
    MSG_CONFIRM_CHANGESET = 'Deploy this changeset?'
    MSG_CONFIRM_CHANGESET_HEADER = '\nPreviewing CloudFormation changeset before deployment'

    def __init__(self, template_file, stack_name, s3_bucket, image_repository, image_repositories, force_upload, no_progressbar, s3_prefix, kms_key_id, parameter_overrides, capabilities, no_execute_changeset, role_arn, notification_arns, fail_on_empty_changeset, tags, region, profile, confirm_changeset, signing_profiles, use_changeset, disable_rollback, poll_delay, on_failure):
        if False:
            for i in range(10):
                print('nop')
        self.template_file = template_file
        self.stack_name = stack_name
        self.s3_bucket = s3_bucket
        self.image_repository = image_repository
        self.image_repositories = image_repositories
        self.force_upload = force_upload
        self.no_progressbar = no_progressbar
        self.s3_prefix = s3_prefix
        self.kms_key_id = kms_key_id
        self.parameter_overrides = parameter_overrides
        self.global_parameter_overrides: Optional[Dict] = None
        if region:
            self.global_parameter_overrides = {IntrinsicsSymbolTable.AWS_REGION: region}
        self.capabilities = capabilities
        self.no_execute_changeset = no_execute_changeset
        self.role_arn = role_arn
        self.notification_arns = notification_arns
        self.fail_on_empty_changeset = fail_on_empty_changeset
        self.tags = tags
        self.region = region
        self.profile = profile
        self.s3_uploader = None
        self.deployer = None
        self.confirm_changeset = confirm_changeset
        self.signing_profiles = signing_profiles
        self.use_changeset = use_changeset
        self.disable_rollback = disable_rollback
        self.poll_delay = poll_delay
        self.on_failure = FailureMode(on_failure) if on_failure else FailureMode.ROLLBACK
        self._max_template_size = 51200

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        pass

    def run(self):
        if False:
            i = 10
            return i + 15
        '\n        Execute deployment based on the argument provided by customers and samconfig.toml.\n        '
        with open(self.template_file, 'r') as handle:
            template_str = handle.read()
        template_dict = yaml_parse(template_str)
        if not isinstance(template_dict, dict):
            raise deploy_exceptions.DeployFailedError(stack_name=self.stack_name, msg='{} not in required format'.format(self.template_file))
        parameters = self.merge_parameters(template_dict, self.parameter_overrides)
        template_size = os.path.getsize(self.template_file)
        if template_size > self._max_template_size and (not self.s3_bucket):
            raise deploy_exceptions.DeployBucketRequiredError()
        boto_config = get_boto_config_with_user_agent()
        cloudformation_client = boto3.client('cloudformation', region_name=self.region if self.region else None, config=boto_config)
        s3_client = None
        if self.s3_bucket:
            s3_client = boto3.client('s3', region_name=self.region if self.region else None, config=boto_config)
            self.s3_uploader = S3Uploader(s3_client, self.s3_bucket, self.s3_prefix, self.kms_key_id, self.force_upload, self.no_progressbar)
        self.deployer = Deployer(cloudformation_client, client_sleep=self.poll_delay)
        region = s3_client._client_config.region_name if s3_client else self.region
        display_parameter_overrides = hide_noecho_parameter_overrides(template_dict, self.parameter_overrides)
        print_deploy_args(self.stack_name, self.s3_bucket, self.image_repositories if isinstance(self.image_repositories, dict) else self.image_repository, region, self.capabilities, display_parameter_overrides, self.confirm_changeset, self.signing_profiles, self.use_changeset, self.disable_rollback)
        return self.deploy(self.stack_name, template_str, parameters, self.capabilities, self.no_execute_changeset, self.role_arn, self.notification_arns, self.s3_uploader, [{'Key': key, 'Value': value} for (key, value) in self.tags.items()] if self.tags else [], region, self.fail_on_empty_changeset, self.confirm_changeset, self.use_changeset, self.disable_rollback)

    def deploy(self, stack_name: str, template_str: str, parameters: List[dict], capabilities: List[str], no_execute_changeset: bool, role_arn: str, notification_arns: List[str], s3_uploader: S3Uploader, tags: List[str], region: str, fail_on_empty_changeset: bool=True, confirm_changeset: bool=False, use_changeset: bool=True, disable_rollback: bool=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Deploy the stack to cloudformation.\n        - if changeset needs confirmation, it will prompt for customers to confirm.\n        - if no_execute_changeset is True, the changeset won't be executed.\n\n        Parameters\n        ----------\n        stack_name : str\n            name of the stack\n        template_str : str\n            the string content of the template\n        parameters : List[Dict]\n            List of parameters\n        capabilities : List[str]\n            List of capabilities\n        no_execute_changeset : bool\n            A bool indicating whether to execute changeset\n        role_arn : str\n            the Arn of the role to create changeset\n        notification_arns : List[str]\n            Arns for sending notifications\n        s3_uploader : S3Uploader\n            S3Uploader object to upload files to S3 buckets\n        tags : List[str]\n            List of tags passed to CloudFormation\n        region : str\n            AWS region to deploy the stack to\n        fail_on_empty_changeset : bool\n            Should fail when changeset is empty\n        confirm_changeset : bool\n            Should wait for customer's confirm before executing the changeset\n        use_changeset : bool\n            Involve creation of changesets, false when using sam sync\n        disable_rollback : bool\n            Preserves the state of previously provisioned resources when an operation fails\n        "
        (stacks, _) = SamLocalStackProvider.get_stacks(self.template_file, parameter_overrides=sanitize_parameter_overrides(self.parameter_overrides), global_parameter_overrides=self.global_parameter_overrides)
        auth_required_per_resource = auth_per_resource(stacks)
        for (resource, authorization_required) in auth_required_per_resource:
            if not authorization_required:
                click.secho(f'{resource} has no authentication.', fg='yellow')
        if use_changeset:
            try:
                (result, changeset_type) = self.deployer.create_and_wait_for_changeset(stack_name=stack_name, cfn_template=template_str, parameter_values=parameters, capabilities=capabilities, role_arn=role_arn, notification_arns=notification_arns, s3_uploader=s3_uploader, tags=tags)
                click.echo(self.MSG_SHOWCASE_CHANGESET.format(changeset_id=result['Id']))
                if no_execute_changeset:
                    return
                if confirm_changeset:
                    click.secho(self.MSG_CONFIRM_CHANGESET_HEADER, fg='yellow')
                    click.secho('=' * len(self.MSG_CONFIRM_CHANGESET_HEADER), fg='yellow')
                    if not click.confirm(f'{self.MSG_CONFIRM_CHANGESET}', default=False):
                        return
                marker_time = self.deployer.get_last_event_time(stack_name, 0)
                self.deployer.execute_changeset(result['Id'], stack_name, disable_rollback)
                self.deployer.wait_for_execute(stack_name, changeset_type, disable_rollback, self.on_failure, marker_time)
                click.echo(self.MSG_EXECUTE_SUCCESS.format(stack_name=stack_name, region=region))
            except deploy_exceptions.ChangeEmptyError as ex:
                if fail_on_empty_changeset:
                    raise
                click.echo(str(ex))
            except deploy_exceptions.DeployFailedError:
                if self.on_failure != FailureMode.DELETE:
                    raise
                self.deployer.rollback_delete_stack(stack_name)
        else:
            try:
                result = self.deployer.sync(stack_name=stack_name, cfn_template=template_str, parameter_values=parameters, capabilities=capabilities, role_arn=role_arn, notification_arns=notification_arns, s3_uploader=s3_uploader, tags=tags, on_failure=self.on_failure)
                LOG.debug(result)
            except deploy_exceptions.DeployFailedError as ex:
                LOG.error(str(ex))
                raise

    @staticmethod
    def merge_parameters(template_dict: Dict, parameter_overrides: Dict) -> List[Dict]:
        if False:
            i = 10
            return i + 15
        '\n        CloudFormation CreateChangeset requires a value for every parameter\n        from the template, either specifying a new value or use previous value.\n        For convenience, this method will accept new parameter values and\n        generates a dict of all parameters in a format that ChangeSet API\n        will accept\n\n        :param template_dict:\n        :param parameter_overrides:\n        :return:\n        '
        parameter_values: List[Dict] = []
        if not isinstance(template_dict.get('Parameters', None), dict):
            return parameter_values
        for (key, _) in template_dict['Parameters'].items():
            obj = {'ParameterKey': key}
            if key in parameter_overrides:
                obj['ParameterValue'] = parameter_overrides[key]
            else:
                obj['UsePreviousValue'] = True
            parameter_values.append(obj)
        return parameter_values