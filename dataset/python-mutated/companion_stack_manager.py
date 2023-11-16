"""
    Companion stack manager
"""
import logging
from typing import Dict, List, Optional
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError
from mypy_boto3_cloudformation.client import CloudFormationClient
from mypy_boto3_cloudformation.type_defs import WaiterConfigTypeDef
from mypy_boto3_s3.client import S3Client
from samcli.commands.exceptions import AWSServiceClientError, RegionError
from samcli.lib.bootstrap.companion_stack.companion_stack_builder import CompanionStackBuilder
from samcli.lib.bootstrap.companion_stack.data_types import CompanionStack, ECRRepo
from samcli.lib.package.artifact_exporter import mktempfile
from samcli.lib.package.s3_uploader import S3Uploader
from samcli.lib.providers.sam_function_provider import SamFunctionProvider
from samcli.lib.providers.sam_stack_provider import SamLocalStackProvider
from samcli.lib.utils.packagetype import IMAGE
from samcli.lib.utils.s3 import parse_s3_url
LOG = logging.getLogger(__name__)

class CompanionStackManager:
    """
    Manager class for a companion stack
    Used to create/update the remote stack
    """
    _companion_stack: CompanionStack
    _builder: CompanionStackBuilder
    _boto_config: Config
    _update_stack_waiter_config: WaiterConfigTypeDef
    _delete_stack_waiter_config: WaiterConfigTypeDef
    _s3_bucket: str
    _s3_prefix: str
    _cfn_client: CloudFormationClient
    _s3_client: S3Client

    def __init__(self, stack_name, region, s3_bucket, s3_prefix):
        if False:
            for i in range(10):
                print('nop')
        self._companion_stack = CompanionStack(stack_name)
        self._builder = CompanionStackBuilder(self._companion_stack)
        self._boto_config = Config(region_name=region if region else None)
        self._update_stack_waiter_config = {'Delay': 10, 'MaxAttempts': 120}
        self._delete_stack_waiter_config = {'Delay': 10, 'MaxAttempts': 120}
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix
        try:
            self._cfn_client = boto3.client('cloudformation', config=self._boto_config)
            self._ecr_client = boto3.client('ecr', config=self._boto_config)
            self._s3_client = boto3.client('s3', config=self._boto_config)
            self._account_id = boto3.client('sts').get_caller_identity().get('Account')
            self._region_name = self._cfn_client.meta.region_name
        except NoCredentialsError as ex:
            raise AWSServiceClientError('Error Setting Up Managed Stack Client: Unable to resolve credentials for the AWS SDK for Python client. Please see their documentation for options to pass in credentials: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html') from ex
        except NoRegionError as ex:
            raise RegionError('Error Setting Up Managed Stack Client: Unable to resolve a region. Please provide a region via the --region parameter or by the AWS_DEFAULT_REGION environment variable.') from ex

    def set_functions(self, function_logical_ids: List[str], image_repositories: Optional[Dict[str, str]]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Sets functions that need to have ECR repos created\n\n        Parameters\n        ----------\n        function_logical_ids: List[str]\n            Function logical IDs that need to have ECR repos created\n        image_repositories: Optional[Dict[str, str]]\n            Optional image repository mapping. Functions with non-auto-ecr URIs\n            will be ignored.\n        '
        self._builder.clear_functions()
        if image_repositories is None:
            image_repositories = dict()
        for function_logical_id in function_logical_ids:
            if function_logical_id not in image_repositories or self.is_repo_uri(image_repositories.get(function_logical_id), function_logical_id):
                self._builder.add_function(function_logical_id)

    def update_companion_stack(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Blocking call to create or update the companion stack based on current functions\n        Companion stack template will be updated to the s3 bucket first before deployment\n        '
        if not self._builder.repo_mapping:
            return
        stack_name = self._companion_stack.stack_name
        template = self._builder.build()
        with mktempfile() as temporary_file:
            temporary_file.write(template)
            temporary_file.flush()
            s3_uploader = S3Uploader(self._s3_client, bucket_name=self._s3_bucket, prefix=self._s3_prefix, no_progressbar=True)
            parts = parse_s3_url(s3_uploader.upload_with_dedup(temporary_file.name, 'template'), version_property='Version')
        template_url = s3_uploader.to_path_style_s3_url(parts['Key'], parts.get('Version', None))
        exists = self.does_companion_stack_exist()
        if exists:
            self._cfn_client.update_stack(StackName=stack_name, TemplateURL=template_url, Capabilities=['CAPABILITY_AUTO_EXPAND'])
            update_waiter = self._cfn_client.get_waiter('stack_update_complete')
            update_waiter.wait(StackName=stack_name, WaiterConfig=self._update_stack_waiter_config)
        else:
            self._cfn_client.create_stack(StackName=stack_name, TemplateURL=template_url, Capabilities=['CAPABILITY_AUTO_EXPAND'])
            create_waiter = self._cfn_client.get_waiter('stack_create_complete')
            create_waiter.wait(StackName=stack_name, WaiterConfig=self._update_stack_waiter_config)

    def _delete_companion_stack(self) -> None:
        if False:
            return 10
        '\n        Blocking call to delete the companion stack\n        '
        stack_name = self._companion_stack.stack_name
        waiter = self._cfn_client.get_waiter('stack_delete_complete')
        self._cfn_client.delete_stack(StackName=stack_name)
        waiter.wait(StackName=stack_name, WaiterConfig=self._delete_stack_waiter_config)

    def list_deployed_repos(self) -> List[ECRRepo]:
        if False:
            return 10
        '\n        List deployed ECR repos for this companion stack\n        Not using create_change_set as it is slow.\n\n        Returns\n        -------\n        List[ECRRepo]\n            List of ECR repos deployed for this companion stack\n            Returns empty list if companion stack does not exist\n        '
        if not self.does_companion_stack_exist():
            return []
        repos: List[ECRRepo] = list()
        stack = boto3.resource('cloudformation', config=self._boto_config).Stack(self._companion_stack.stack_name)
        for resource in stack.resource_summaries.all():
            if resource.resource_type == 'AWS::ECR::Repository':
                repos.append(ECRRepo(logical_id=resource.logical_resource_id, physical_id=resource.physical_resource_id))
        return repos

    def get_unreferenced_repos(self) -> List[ECRRepo]:
        if False:
            for i in range(10):
                print('nop')
        '\n        List deployed ECR repos that is not referenced by current list of functions\n\n        Returns\n        -------\n        List[ECRRepo]\n            List of deployed ECR repos that is not referenced by current list of functions\n            Returns empty list if companion stack does not exist\n        '
        if not self.does_companion_stack_exist():
            return []
        deployed_repos: List[ECRRepo] = self.list_deployed_repos()
        current_mapping = self._builder.repo_mapping
        unreferenced_repos: List[ECRRepo] = list()
        for deployed_repo in deployed_repos:
            for (_, current_repo) in current_mapping.items():
                if current_repo.logical_id == deployed_repo.logical_id:
                    break
            else:
                unreferenced_repos.append(deployed_repo)
        return unreferenced_repos

    def delete_unreferenced_repos(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Blocking call to delete all deployed ECR repos that are unreferenced by a function\n        If repo does not exist, this will simply skip it.\n        '
        repos = self.get_unreferenced_repos()
        for repo in repos:
            try:
                self._ecr_client.delete_repository(repositoryName=repo.physical_id, force=True)
            except self._ecr_client.exceptions.RepositoryNotFoundException:
                LOG.debug('Image repo [%s] not found in companion stack. Skipping deletion.', repo.physical_id)

    def sync_repos(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Blocking call to sync companion stack with the following actions\n        Creates the stack if it does not exist, and updates it if it does.\n        Deletes unreferenced repos if they exist.\n        Deletes companion stack if there isn't any repo left.\n        "
        has_repo = bool(self.get_repository_mapping())
        if self.does_companion_stack_exist():
            self.delete_unreferenced_repos()
            if has_repo:
                self.update_companion_stack()
            else:
                self._delete_companion_stack()
        elif has_repo:
            self.update_companion_stack()

    def does_companion_stack_exist(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Does companion stack exist\n\n        Returns\n        -------\n        bool\n            Returns True if companion stack exists\n        '
        try:
            self._cfn_client.describe_stacks(StackName=self._companion_stack.stack_name)
            return True
        except ClientError as e:
            error_message = e.response.get('Error', {}).get('Message')
            if error_message == f'Stack with id {self._companion_stack.stack_name} does not exist':
                return False
            raise e

    def get_repository_mapping(self) -> Dict[str, str]:
        if False:
            print('Hello World!')
        '\n        Get current function to repo mapping\n\n        Returns\n        -------\n        Dict[str, str]\n            Dictionary with key as function logical ID and value as ECR repo URI.\n        '
        return dict(((k, self.get_repo_uri(v)) for (k, v) in self._builder.repo_mapping.items()))

    def get_repo_uri(self, repo: ECRRepo) -> str:
        if False:
            while True:
                i = 10
        '\n        Get repo URI for a ECR repo\n\n        Parameters\n        ----------\n        repo: ECRRepo\n\n        Returns\n        -------\n        str\n            ECR repo URI based on account ID and region.\n        '
        return repo.get_repo_uri(self._account_id, self._region_name)

    def is_repo_uri(self, repo_uri: Optional[str], function_logical_id: str) -> bool:
        if False:
            return 10
        '\n        Check whether repo URI is a companion stack repo\n\n        Parameters\n        ----------\n        repo_uri: str\n            Repo URI to be checked.\n\n        function_logical_id: str\n            Function logical ID associated with the image repo.\n\n        Returns\n        -------\n        bool\n            Returns True if repo_uri is a companion stack repo.\n        '
        return repo_uri == self.get_repo_uri(ECRRepo(self._companion_stack, function_logical_id))

def sync_ecr_stack(template_file: str, stack_name: str, region: str, s3_bucket: str, s3_prefix: str, image_repositories: Dict[str, str]) -> Dict[str, str]:
    if False:
        i = 10
        return i + 15
    'Blocking call to sync local functions with ECR Companion Stack\n\n    Parameters\n    ----------\n    template_file : str\n        Template file path.\n    stack_name : str\n        Stack name\n    region : str\n        AWS region\n    s3_bucket : str\n        S3 bucket\n    s3_prefix : str\n        S3 prefix for the bucket\n    image_repositories : Dict[str, str]\n        Mapping between function logical ID and ECR URI\n\n    Returns\n    -------\n    Dict[str, str]\n        Updated mapping of image_repositories. Auto ECR URIs are added\n        for Functions without a repo specified.\n    '
    image_repositories = image_repositories.copy() if image_repositories else {}
    manager = CompanionStackManager(stack_name, region, s3_bucket, s3_prefix)
    stacks = SamLocalStackProvider.get_stacks(template_file)[0]
    function_provider = SamFunctionProvider(stacks, ignore_code_extraction_warnings=True)
    function_logical_ids = [function.full_path for function in function_provider.get_all() if function.packagetype == IMAGE]
    manager.set_functions(function_logical_ids, image_repositories)
    image_repositories.update(manager.get_repository_mapping())
    manager.sync_repos()
    return image_repositories