"""Lambda models for internal use and persistence.
The LambdaProviderPro in localstack-ext imports this model and configures persistence.
The actual function code is stored in S3 (see S3Code).
"""
import dataclasses
import logging
import shutil
import tempfile
import threading
from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import IO, Dict, Literal, Optional, TypedDict
from botocore.exceptions import ClientError
from localstack import config
from localstack.aws.api import CommonServiceException
from localstack.aws.api.lambda_ import AllowedPublishers, Architecture, CodeSigningPolicies, Cors, DestinationConfig, FunctionUrlAuthType, InvocationType, LastUpdateStatus, PackageType, ProvisionedConcurrencyStatusEnum, Runtime, RuntimeVersionConfig, SnapStartResponse, State, StateReasonCode, TracingMode
from localstack.aws.connect import connect_to
from localstack.constants import AWS_REGION_US_EAST_1
from localstack.services.lambda_.api_utils import qualified_lambda_arn, unqualified_lambda_arn
from localstack.utils.archives import unzip
from localstack.utils.strings import long_uid
LOG = logging.getLogger(__name__)
IMAGE_MAPPING = {'python3.7': 'python:3.7', 'python3.8': 'python:3.8', 'python3.9': 'python:3.9', 'python3.10': 'python:3.10', 'python3.11': 'python:3.11', 'nodejs12.x': 'nodejs:12', 'nodejs14.x': 'nodejs:14', 'nodejs16.x': 'nodejs:16', 'nodejs18.x': 'nodejs:18', 'ruby2.7': 'ruby:2.7', 'ruby3.2': 'ruby:3.2', 'java8': 'java:8', 'java8.al2': 'java:8.al2', 'java11': 'java:11', 'java17': 'java:17', 'dotnetcore3.1': 'dotnet:core3.1', 'dotnet6': 'dotnet:6', 'go1.x': 'go:1', 'provided': 'provided:alami', 'provided.al2': 'provided:al2'}
SNAP_START_SUPPORTED_RUNTIMES = [Runtime.java11, Runtime.java17]

@dataclasses.dataclass(frozen=True)
class VersionState:
    state: State
    code: Optional[StateReasonCode] = None
    reason: Optional[str] = None

@dataclasses.dataclass
class Invocation:
    payload: bytes
    invoked_arn: str
    client_context: Optional[str]
    invocation_type: InvocationType
    invoke_time: datetime
    request_id: str
InitializationType = Literal['on-demand', 'provisioned-concurrency']

class ArchiveCode(metaclass=ABCMeta):

    @abstractmethod
    def generate_presigned_url(self, endpoint_url: str | None=None):
        if False:
            i = 10
            return i + 15
        '\n        Generates a presigned url pointing to the code archive\n        '
        pass

    @abstractmethod
    def is_hot_reloading(self):
        if False:
            i = 10
            return i + 15
        '\n        Whether this code archive is for hot reloading.\n        This means it should mount the location from the host, and should instruct the runtimes to listen for changes\n\n        :return: True if this object represents hot reloading, False otherwise\n        '
        pass

    @abstractmethod
    def get_unzipped_code_location(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the location of the unzipped archive on disk\n        '
        pass

    @abstractmethod
    def prepare_for_execution(self):
        if False:
            print('Hello World!')
        '\n        Unzips the code archive to the proper destination on disk, if not already present\n        '
        pass

    @abstractmethod
    def destroy_cached(self):
        if False:
            while True:
                i = 10
        '\n        Destroys the code object on disk, if it was saved on disk before\n        '
        pass

    @abstractmethod
    def destroy(self):
        if False:
            print('Hello World!')
        '\n        Deletes the code object from S3 and the unzipped version from disk\n        '
        pass

@dataclasses.dataclass(frozen=True)
class S3Code(ArchiveCode):
    """
    Objects representing a code archive stored in an internal S3 bucket.

    S3 Store:
      Code archives represented by this method are stored in a bucket awslambda-{region_name}-tasks,
      (e.g. awslambda-us-east-1-tasks), when correctly created using create_lambda_archive.
      The "awslambda" prefix matches the behavior at real AWS.

      This class will then provide different properties / methods to be operated on the stored code,
      like the ability to create presigned-urls, checking the code hash etc.

      A call to destroy() of this class will delete the code object from both the S3 store and the local cache
    Unzipped Cache:
      After a call to prepare_for_execution, an unzipped version of the represented code will be stored on disk,
      ready to mount/copy.

      It will be present at the location returned by get_unzipped_code_location,
      namely /tmp/lambda/{bucket_name}/{id}/code

      The cache on disk will be deleted after a call to destroy_cached (or destroy)
    """
    id: str
    account_id: str
    s3_bucket: str
    s3_key: str
    s3_object_version: str | None
    code_sha256: str
    code_size: int
    _disk_lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

    def _download_archive_to_file(self, target_file: IO) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Download the code archive into a given file\n\n        :param target_file: File the code archive should be downloaded into (IO object)\n        '
        s3_client = connect_to(region_name=AWS_REGION_US_EAST_1, aws_access_key_id=config.INTERNAL_RESOURCE_ACCOUNT).s3
        extra_args = {'VersionId': self.s3_object_version} if self.s3_object_version else {}
        s3_client.download_fileobj(Bucket=self.s3_bucket, Key=self.s3_key, Fileobj=target_file, ExtraArgs=extra_args)
        target_file.flush()

    def generate_presigned_url(self, endpoint_url: str | None=None) -> str:
        if False:
            while True:
                i = 10
        '\n        Generates a presigned url pointing to the code archive\n        '
        s3_client = connect_to(region_name=AWS_REGION_US_EAST_1, aws_access_key_id=config.INTERNAL_RESOURCE_ACCOUNT, endpoint_url=endpoint_url).s3
        params = {'Bucket': self.s3_bucket, 'Key': self.s3_key}
        if self.s3_object_version:
            params['VersionId'] = self.s3_object_version
        return s3_client.generate_presigned_url('get_object', Params=params)

    def is_hot_reloading(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Whether this code archive is hot reloading\n\n        :return: True if it must it represents hot reloading, False otherwise\n        '
        return False

    def get_unzipped_code_location(self) -> Path:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the location of the unzipped archive on disk\n        '
        return Path(f'{tempfile.gettempdir()}/lambda/{self.s3_bucket}/{self.id}/code')

    def prepare_for_execution(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Unzips the code archive to the proper destination on disk, if not already present\n        '
        target_path = self.get_unzipped_code_location()
        with self._disk_lock:
            if target_path.exists():
                return
            LOG.debug('Saving code %s to disk', self.id)
            target_path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile() as file:
                self._download_archive_to_file(file)
                unzip(file.name, str(target_path))

    def destroy_cached(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Destroys the code object on disk, if it was saved on disk before\n        '
        code_path = self.get_unzipped_code_location().parent
        if not code_path.exists():
            return
        try:
            shutil.rmtree(code_path)
        except OSError as e:
            LOG.debug('Could not cleanup function code path %s due to error %s while deleting file %s', code_path, e.strerror, e.filename)

    def destroy(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Deletes the code object from S3 and the unzipped version from disk\n        '
        LOG.debug('Final code destruction for %s', self.id)
        self.destroy_cached()
        s3_client = connect_to(region_name=AWS_REGION_US_EAST_1, aws_access_key_id=config.INTERNAL_RESOURCE_ACCOUNT).s3
        kwargs = {'VersionId': self.s3_object_version} if self.s3_object_version else {}
        try:
            s3_client.delete_object(Bucket=self.s3_bucket, Key=self.s3_key, **kwargs)
        except ClientError as e:
            LOG.debug('Cannot delete lambda archive %s in bucket %s: %s', self.s3_key, self.s3_bucket, e)

@dataclasses.dataclass(frozen=True)
class HotReloadingCode(ArchiveCode):
    """
    Objects representing code which is mounted from a given directory from the host, for hot reloading
    """
    host_path: str
    code_sha256: str = 'hot-reloading-hash-not-available'
    code_size: int = 0

    def generate_presigned_url(self, endpoint_url: str | None=None) -> str:
        if False:
            while True:
                i = 10
        return f'Code location: {self.host_path}'

    def get_unzipped_code_location(self) -> Path:
        if False:
            print('Hello World!')
        return Path(self.host_path)

    def is_hot_reloading(self) -> bool:
        if False:
            return 10
        '\n        Whether this code archive is for hot reloading.\n        This means it should mount the location from the host, and should instruct the runtimes to listen for changes\n\n        :return: True if it represents hot reloading, False otherwise\n        '
        return True

    def prepare_for_execution(self) -> None:
        if False:
            print('Hello World!')
        pass

    def destroy_cached(self) -> None:
        if False:
            print('Hello World!')
        '\n        Destroys the code object on disk, if it was saved on disk before\n        '
        pass

    def destroy(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Deletes the code object from S3 and the unzipped version from disk\n        '
        pass

@dataclasses.dataclass(frozen=True)
class ImageCode:
    image_uri: str
    repository_type: str
    code_sha256: str

    @property
    def resolved_image_uri(self):
        if False:
            for i in range(10):
                print('nop')
        return f"{self.image_uri.rpartition(':')[0]}@sha256:{self.code_sha256}"

@dataclasses.dataclass
class DeadLetterConfig:
    target_arn: str

@dataclasses.dataclass
class FileSystemConfig:
    arn: str
    local_mount_path: str

@dataclasses.dataclass(frozen=True)
class ImageConfig:
    working_directory: str
    command: list[str] = dataclasses.field(default_factory=list)
    entrypoint: list[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class VpcConfig:
    vpc_id: str
    security_group_ids: list[str] = dataclasses.field(default_factory=list)
    subnet_ids: list[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass(frozen=True)
class UpdateStatus:
    status: LastUpdateStatus | None
    code: str | None = None
    reason: str | None = None

@dataclasses.dataclass
class LambdaEphemeralStorage:
    size: int

@dataclasses.dataclass
class FunctionUrlConfig:
    """
    * HTTP(s)
    * You can apply function URLs to any function alias, or to the $LATEST unpublished function version. You can't add a function URL to any other function version.
    * Once you create a function URL, its URL endpoint never changes
    """
    function_arn: str
    function_name: str
    cors: Cors
    url_id: str
    url: str
    auth_type: FunctionUrlAuthType
    creation_time: str
    last_modified_time: Optional[str] = None
    function_qualifier: Optional[str] = '$LATEST'

@dataclasses.dataclass
class ProvisionedConcurrencyConfiguration:
    provisioned_concurrent_executions: int
    last_modified: str

@dataclasses.dataclass
class ProvisionedConcurrencyState:
    """transient items"""
    allocated: int = 0
    available: int = 0
    status: ProvisionedConcurrencyStatusEnum = dataclasses.field(default=ProvisionedConcurrencyStatusEnum.IN_PROGRESS)
    status_reason: Optional[str] = None

@dataclasses.dataclass
class AliasRoutingConfig:
    version_weights: Dict[str, float]

@dataclasses.dataclass(frozen=True)
class VersionIdentifier:
    function_name: str
    qualifier: str
    region: str
    account: str

    def qualified_arn(self):
        if False:
            i = 10
            return i + 15
        return qualified_lambda_arn(function_name=self.function_name, qualifier=self.qualifier, region=self.region, account=self.account)

    def unqualified_arn(self):
        if False:
            while True:
                i = 10
        return unqualified_lambda_arn(function_name=self.function_name, region=self.region, account=self.account)

@dataclasses.dataclass(frozen=True)
class VersionAlias:
    function_version: str
    name: str
    description: str | None
    routing_configuration: AliasRoutingConfig | None = None
    revision_id: str = dataclasses.field(init=False, default_factory=long_uid)

@dataclasses.dataclass
class ResourcePolicy:
    Version: str
    Id: str
    Statement: list[dict]

@dataclasses.dataclass
class FunctionResourcePolicy:
    policy: ResourcePolicy

@dataclasses.dataclass
class EventInvokeConfig:
    function_name: str
    qualifier: str
    last_modified: Optional[str] = dataclasses.field(compare=False)
    destination_config: Optional[DestinationConfig] = None
    maximum_retry_attempts: Optional[int] = None
    maximum_event_age_in_seconds: Optional[int] = None

@dataclasses.dataclass
class InvocationResult:
    request_id: str
    payload: bytes | None
    is_error: bool
    logs: str | None
    executed_version: str | None = None

@dataclasses.dataclass
class InvocationLogs:
    request_id: str
    logs: str

class Credentials(TypedDict):
    AccessKeyId: str
    SecretAccessKey: str
    SessionToken: str
    Expiration: datetime

class OtherServiceEndpoint:

    def status_ready(self, executor_id: str) -> None:
        if False:
            print('Hello World!')
        '\n        Processes a status ready report by RAPID\n        :param executor_id: Executor ID this ready report is for\n        '
        raise NotImplementedError()

    def status_error(self, executor_id: str) -> None:
        if False:
            return 10
        '\n        Processes a status error report by RAPID\n        :param executor_id: Executor ID this error report is for\n        '
        raise NotImplementedError()

@dataclasses.dataclass(frozen=True)
class CodeSigningConfig:
    csc_id: str
    arn: str
    allowed_publishers: AllowedPublishers
    policies: CodeSigningPolicies
    last_modified: str
    description: Optional[str] = None

@dataclasses.dataclass
class LayerPolicyStatement:
    sid: str
    action: str
    principal: str
    organization_id: Optional[str]

@dataclasses.dataclass
class LayerPolicy:
    revision_id: str = dataclasses.field(init=False, default_factory=long_uid)
    id: str = 'default'
    version: str = '2012-10-17'
    statements: dict[str, LayerPolicyStatement] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class LayerVersion:
    layer_version_arn: str
    layer_arn: str
    version: int
    code: ArchiveCode
    license_info: str
    compatible_runtimes: list[Runtime]
    compatible_architectures: list[Architecture]
    created: str
    description: str = ''
    policy: LayerPolicy = None

@dataclasses.dataclass
class Layer:
    arn: str
    next_version: int = 1
    next_version_lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    layer_versions: dict[str, LayerVersion] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass(frozen=True)
class VersionFunctionConfiguration:
    description: str
    role: str
    timeout: int
    runtime: Runtime
    memory_size: int
    handler: str
    package_type: PackageType
    environment: dict[str, str]
    architectures: list[Architecture]
    internal_revision: str
    ephemeral_storage: LambdaEphemeralStorage
    snap_start: SnapStartResponse
    tracing_config_mode: TracingMode
    code: ArchiveCode
    last_modified: str
    state: VersionState
    image: Optional[ImageCode] = None
    image_config: Optional[ImageConfig] = None
    runtime_version_config: Optional[RuntimeVersionConfig] = None
    last_update: Optional[UpdateStatus] = None
    revision_id: str = dataclasses.field(init=False, default_factory=long_uid)
    layers: list[LayerVersion] = dataclasses.field(default_factory=list)
    dead_letter_arn: Optional[str] = None
    vpc_config: Optional[VpcConfig] = None

@dataclasses.dataclass(frozen=True)
class FunctionVersion:
    id: VersionIdentifier
    config: VersionFunctionConfiguration

    @property
    def qualified_arn(self) -> str:
        if False:
            return 10
        return self.id.qualified_arn()

@dataclasses.dataclass
class Function:
    function_name: str
    code_signing_config_arn: Optional[str] = None
    aliases: dict[str, VersionAlias] = dataclasses.field(default_factory=dict)
    versions: dict[str, FunctionVersion] = dataclasses.field(default_factory=dict)
    function_url_configs: dict[str, FunctionUrlConfig] = dataclasses.field(default_factory=dict)
    permissions: dict[str, FunctionResourcePolicy] = dataclasses.field(default_factory=dict)
    event_invoke_configs: dict[str, EventInvokeConfig] = dataclasses.field(default_factory=dict)
    reserved_concurrent_executions: Optional[int] = None
    provisioned_concurrency_configs: dict[str, ProvisionedConcurrencyConfiguration] = dataclasses.field(default_factory=dict)
    tags: dict[str, str] | None = None
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    next_version: int = 1

    def latest(self) -> FunctionVersion:
        if False:
            for i in range(10):
                print('nop')
        return self.versions['$LATEST']

class ValidationException(CommonServiceException):

    def __init__(self, message: str):
        if False:
            while True:
                i = 10
        super().__init__(code='ValidationException', status_code=400, message=message)

class RequestEntityTooLargeException(CommonServiceException):

    def __init__(self, message: str):
        if False:
            print('Hello World!')
        super().__init__(code='RequestEntityTooLargeException', status_code=413, message=message)

@dataclasses.dataclass
class AccountSettings:
    total_code_size: int = config.LAMBDA_LIMITS_TOTAL_CODE_SIZE
    code_size_zipped: int = config.LAMBDA_LIMITS_CODE_SIZE_ZIPPED
    code_size_unzipped: int = config.LAMBDA_LIMITS_CODE_SIZE_UNZIPPED
    concurrent_executions: int = config.LAMBDA_LIMITS_CONCURRENT_EXECUTIONS