import base64
import concurrent.futures
import dataclasses
import io
import logging
import random
import uuid
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
from pathlib import PurePosixPath, PureWindowsPath
from threading import RLock
from typing import TYPE_CHECKING, Optional
from localstack import config
from localstack.aws.api.lambda_ import InvalidParameterValueException, InvalidRequestContentException, InvocationType, LastUpdateStatus, ResourceConflictException, ResourceNotFoundException, State
from localstack.aws.connect import connect_to
from localstack.constants import AWS_REGION_US_EAST_1
from localstack.services.lambda_ import usage
from localstack.services.lambda_.api_utils import lambda_arn, qualified_lambda_arn, qualifier_is_alias
from localstack.services.lambda_.invocation.assignment import AssignmentService
from localstack.services.lambda_.invocation.counting_service import CountingService
from localstack.services.lambda_.invocation.event_manager import LambdaEventManager
from localstack.services.lambda_.invocation.lambda_models import ArchiveCode, Function, FunctionVersion, HotReloadingCode, ImageCode, Invocation, InvocationResult, S3Code, UpdateStatus, VersionAlias, VersionState
from localstack.services.lambda_.invocation.models import lambda_stores
from localstack.services.lambda_.invocation.version_manager import LambdaVersionManager
from localstack.services.lambda_.lambda_utils import HINT_LOG
from localstack.utils.archives import get_unzipped_size, is_zip_file
from localstack.utils.container_utils.container_client import ContainerException
from localstack.utils.docker_utils import DOCKER_CLIENT as CONTAINER_CLIENT
from localstack.utils.strings import short_uid, to_str
if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
LOG = logging.getLogger(__name__)
LAMBDA_DEFAULT_TIMEOUT_SECONDS = 3
LAMBDA_DEFAULT_MEMORY_SIZE = 128

class LambdaService:
    lambda_running_versions: dict[str, LambdaVersionManager]
    lambda_starting_versions: dict[str, LambdaVersionManager]
    event_managers = dict[str, LambdaEventManager]
    lambda_version_manager_lock: RLock
    task_executor: Executor
    assignment_service: AssignmentService
    counting_service: CountingService

    def __init__(self) -> None:
        if False:
            return 10
        self.lambda_running_versions = {}
        self.lambda_starting_versions = {}
        self.event_managers = {}
        self.lambda_version_manager_lock = RLock()
        self.task_executor = ThreadPoolExecutor(thread_name_prefix='lambda-service-task')
        self.assignment_service = AssignmentService()
        self.counting_service = CountingService()

    def stop(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Stop the whole lambda service\n        '
        shutdown_futures = []
        for event_manager in self.event_managers.values():
            shutdown_futures.append(self.task_executor.submit(event_manager.stop))
        for version_manager in self.lambda_running_versions.values():
            shutdown_futures.append(self.task_executor.submit(version_manager.stop))
        for version_manager in self.lambda_starting_versions.values():
            shutdown_futures.append(self.task_executor.submit(version_manager.stop))
            shutdown_futures.append(self.task_executor.submit(version_manager.function_version.config.code.destroy_cached))
        (_, not_done) = concurrent.futures.wait(shutdown_futures, timeout=5)
        if not_done:
            LOG.debug('Shutdown not complete, missing threads: %s', not_done)
        self.task_executor.shutdown(cancel_futures=True)
        self.assignment_service.stop()

    def stop_version(self, qualified_arn: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Stops a specific lambda service version\n        :param qualified_arn: Qualified arn for the version to stop\n        '
        LOG.debug('Stopping version %s', qualified_arn)
        event_manager = self.event_managers.pop(qualified_arn, None)
        if not event_manager:
            LOG.debug('Could not find event manager to stop for function %s...', qualified_arn)
        else:
            self.task_executor.submit(event_manager.stop)
        version_manager = self.lambda_running_versions.pop(qualified_arn, self.lambda_starting_versions.pop(qualified_arn, None))
        if not version_manager:
            raise ValueError(f'Unable to find version manager for {qualified_arn}')
        self.task_executor.submit(version_manager.stop)

    def get_lambda_version_manager(self, function_arn: str) -> LambdaVersionManager:
        if False:
            i = 10
            return i + 15
        '\n        Get the lambda version for the given arn\n        :param function_arn: qualified arn for the lambda version\n        :return: LambdaVersionManager for the arn\n        '
        version_manager = self.lambda_running_versions.get(function_arn)
        if not version_manager:
            raise ValueError(f"Could not find version '{function_arn}'. Is it created?")
        return version_manager

    def get_lambda_event_manager(self, function_arn: str) -> LambdaEventManager:
        if False:
            while True:
                i = 10
        '\n        Get the lambda event manager for the given arn\n        :param function_arn: qualified arn for the lambda version\n        :return: LambdaEventManager for the arn\n        '
        event_manager = self.event_managers.get(function_arn)
        if not event_manager:
            raise ValueError(f"Could not find event manager '{function_arn}'. Is it created?")
        return event_manager

    def create_function_version(self, function_version: FunctionVersion) -> Future[None]:
        if False:
            while True:
                i = 10
        '\n        Creates a new function version (manager), and puts it in the startup dict\n\n        :param function_version: Function Version to create\n        '
        with self.lambda_version_manager_lock:
            qualified_arn = function_version.id.qualified_arn()
            version_manager = self.lambda_starting_versions.get(qualified_arn)
            if version_manager:
                raise Exception("Version '%s' already starting up and in state %s", qualified_arn, version_manager.state)
            state = lambda_stores[function_version.id.account][function_version.id.region]
            fn = state.functions.get(function_version.id.function_name)
            version_manager = LambdaVersionManager(function_arn=qualified_arn, function_version=function_version, lambda_service=self, function=fn, counting_service=self.counting_service, assignment_service=self.assignment_service)
            self.lambda_starting_versions[qualified_arn] = version_manager
        return self.task_executor.submit(version_manager.start)

    def publish_version(self, function_version: FunctionVersion):
        if False:
            print('Hello World!')
        '\n        Synchronously create a function version (manager)\n        Should only be called on publishing new versions, which basically clone an existing one.\n        The new version needs to be added to the lambda store before invoking this.\n        After successful completion of this method, the lambda version stored will be modified to be active, with a new revision id.\n        It will then be active for execution, and should be retrieved again from the store before returning the data over the API.\n\n        :param function_version: Function Version to create\n        '
        with self.lambda_version_manager_lock:
            qualified_arn = function_version.id.qualified_arn()
            version_manager = self.lambda_starting_versions.get(qualified_arn)
            if version_manager:
                raise Exception("Version '%s' already starting up and in state %s", qualified_arn, version_manager.state)
            state = lambda_stores[function_version.id.account][function_version.id.region]
            fn = state.functions.get(function_version.id.function_name)
            version_manager = LambdaVersionManager(function_arn=qualified_arn, function_version=function_version, lambda_service=self, function=fn, counting_service=self.counting_service, assignment_service=self.assignment_service)
            self.lambda_starting_versions[qualified_arn] = version_manager
        version_manager.start()

    def invoke(self, function_name: str, qualifier: str, region: str, account_id: str, invocation_type: InvocationType | None, client_context: Optional[str], request_id: str, payload: bytes | None) -> InvocationResult | None:
        if False:
            i = 10
            return i + 15
        '\n        Invokes a specific version of a lambda\n\n        :param request_id: context request ID\n        :param function_name: Function name\n        :param qualifier: Function version qualifier\n        :param region: Region of the function\n        :param account_id: Account id of the function\n        :param invocation_type: Invocation Type\n        :param client_context: Client Context, if applicable\n        :param payload: Invocation payload\n        :return: The invocation result\n        '
        invoked_arn = lambda_arn(function_name=function_name, qualifier=qualifier, account=account_id, region=region)
        qualifier = qualifier or '$LATEST'
        state = lambda_stores[account_id][region]
        function = state.functions.get(function_name)
        if function is None:
            raise ResourceNotFoundException(f'Function not found: {invoked_arn}', Type='User')
        if qualifier_is_alias(qualifier):
            alias = function.aliases.get(qualifier)
            if not alias:
                raise ResourceNotFoundException(f'Function not found: {invoked_arn}', Type='User')
            version_qualifier = alias.function_version
            if alias.routing_configuration:
                (version, probability) = next(iter(alias.routing_configuration.version_weights.items()))
                if random.random() < probability:
                    version_qualifier = version
        else:
            version_qualifier = qualifier
        qualified_arn = qualified_lambda_arn(function_name, version_qualifier, account_id, region)
        try:
            version_manager = self.get_lambda_version_manager(qualified_arn)
            event_manager = self.get_lambda_event_manager(qualified_arn)
            usage.runtime.record(version_manager.function_version.config.runtime)
        except ValueError as e:
            version = function.versions.get(version_qualifier)
            state = version and version.config.state.state
            if state == State.Failed:
                HINT_LOG.error(f'Failed to create the runtime executor for the function {function_name}. Please ensure that Docker is available in the LocalStack container by adding the volume mount "/var/run/docker.sock:/var/run/docker.sock" to your LocalStack startup. Check out https://docs.localstack.cloud/user-guide/aws/lambda/#docker-not-available')
            elif state == State.Pending:
                HINT_LOG.warning(f'Lambda functions are created and updated asynchronously in the new lambda provider like in AWS. Before invoking {function_name}, please wait until the function transitioned from the state Pending to Active using: "awslocal lambda wait function-active-v2 --function-name {function_name}" Check out https://docs.localstack.cloud/user-guide/aws/lambda/#function-in-pending-state')
            raise ResourceConflictException(f'The operation cannot be performed at this time. The function is currently in the following state: {state}') from e
        if payload is None:
            payload = b'{}'
        else:
            try:
                to_str(payload)
            except Exception as e:
                raise InvalidRequestContentException(f'Could not parse request body into json: Could not parse payload into json: {e}', Type='User')
        if invocation_type is None:
            invocation_type = InvocationType.RequestResponse
        if invocation_type == InvocationType.DryRun:
            return None
        if invocation_type == InvocationType.Event:
            return event_manager.enqueue_event(invocation=Invocation(payload=payload, invoked_arn=invoked_arn, client_context=client_context, invocation_type=invocation_type, invoke_time=datetime.now(), request_id=request_id))
        return version_manager.invoke(invocation=Invocation(payload=payload, invoked_arn=invoked_arn, client_context=client_context, invocation_type=invocation_type, invoke_time=datetime.now(), request_id=request_id))

    def update_version(self, new_version: FunctionVersion) -> Future[None]:
        if False:
            print('Hello World!')
        '\n        Updates a given version. Will perform a rollover, so the old version will be active until the new one is ready\n        to be invoked\n\n        :param new_version: New version (with the same qualifier as an older one)\n        '
        if new_version.qualified_arn not in self.lambda_running_versions:
            raise ValueError(f'Version {new_version.qualified_arn} cannot be updated if an old one is not running')
        return self.create_function_version(function_version=new_version)

    def update_version_state(self, function_version: FunctionVersion, new_state: VersionState) -> None:
        if False:
            return 10
        '\n        Update the version state for the given function version.\n\n        This will perform a rollover to the given function if the new state is active and there is a previously\n        running version registered. The old version will be shutdown and its code deleted.\n\n        If the new state is failed, it will abort the update and mark it as failed.\n        If an older version is still running, it will keep running.\n\n        :param function_version: Version reporting the state\n        :param new_state: New state\n        '
        function_arn = function_version.qualified_arn
        try:
            old_version = None
            old_event_manager = None
            with self.lambda_version_manager_lock:
                new_version_manager = self.lambda_starting_versions.pop(function_arn)
                if not new_version_manager:
                    raise ValueError(f'Version {function_arn} reporting state {new_state.state} does exist in the starting versions.')
                if new_state.state == State.Active:
                    old_version = self.lambda_running_versions.get(function_arn, None)
                    old_event_manager = self.event_managers.get(function_arn, None)
                    self.lambda_running_versions[function_arn] = new_version_manager
                    self.event_managers[function_arn] = LambdaEventManager(version_manager=new_version_manager)
                    self.event_managers[function_arn].start()
                    update_status = UpdateStatus(status=LastUpdateStatus.Successful)
                elif new_state.state == State.Failed:
                    update_status = UpdateStatus(status=LastUpdateStatus.Failed)
                    self.task_executor.submit(new_version_manager.stop)
                else:
                    self.task_executor.submit(new_version_manager.stop)
                    LOG.error('State %s for version %s should not have been reported. New version will be stopped.', new_state, function_arn)
                    return
            state = lambda_stores[function_version.id.account][function_version.id.region]
            function = state.functions.get(function_version.id.function_name)
            if old_event_manager:
                self.task_executor.submit(old_event_manager.stop_for_update)
            if old_version:
                self.task_executor.submit(old_version.stop)
                if function:
                    self.task_executor.submit(destroy_code_if_not_used, old_version.function_version.config.code, function)
            if not function:
                LOG.debug('Function %s was deleted during status update', function_arn)
                return
            current_version = function.versions[function_version.id.qualifier]
            new_version_manager.state = new_state
            new_version_state = dataclasses.replace(current_version, config=dataclasses.replace(current_version.config, state=new_state, last_update=update_status))
            state.functions[function_version.id.function_name].versions[function_version.id.qualifier] = new_version_state
        except Exception:
            LOG.exception('Failed to update function version for arn %s', function_arn)

    def update_alias(self, old_alias: VersionAlias, new_alias: VersionAlias, function: Function):
        if False:
            for i in range(10):
                print('nop')
        provisioned_concurrency_config = function.provisioned_concurrency_configs.get(old_alias.name)
        if old_alias.function_version != new_alias.function_version and provisioned_concurrency_config is not None:
            LOG.warning('Deprovisioning')
            fn_version_old = function.versions.get(old_alias.function_version)
            vm_old = self.get_lambda_version_manager(function_arn=fn_version_old.qualified_arn)
            fn_version_new = function.versions.get(new_alias.function_version)
            vm_new = self.get_lambda_version_manager(function_arn=fn_version_new.qualified_arn)
            vm_old.update_provisioned_concurrency_config(0).result(timeout=4)
            vm_new.update_provisioned_concurrency_config(provisioned_concurrency_config.provisioned_concurrent_executions)

    def can_assume_role(self, role_arn: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks whether lambda can assume the given role.\n        This _should_ only fail if IAM enforcement is enabled.\n\n        :param role_arn: Role to assume\n        :return: True if the role can be assumed by lambda, false otherwise\n        '
        sts_client = connect_to().sts.request_metadata(service_principal='lambda')
        try:
            sts_client.assume_role(RoleArn=role_arn, RoleSessionName=f'test-assume-{short_uid()}', DurationSeconds=900)
            return True
        except Exception as e:
            LOG.debug('Cannot assume role %s: %s', role_arn, e)
            return False

def is_code_used(code: S3Code, function: Function) -> bool:
    if False:
        while True:
            i = 10
    '\n    Check if given code is still used in some version of the function\n\n    :param code: Code object\n    :param function: function to check\n    :return: bool whether code is used in another version of the function\n    '
    with function.lock:
        return any((code == version.config.code for version in function.versions.values()))

def destroy_code_if_not_used(code: S3Code, function: Function) -> None:
    if False:
        while True:
            i = 10
    '\n    Destroy the given code if it is not used in some version of the function\n    Do nothing otherwise\n\n    :param code: Code object\n    :param function: Function the code belongs too\n    '
    with function.lock:
        if not is_code_used(code, function):
            code.destroy()

def store_lambda_archive(archive_file: bytes, function_name: str, region_name: str, account_id: str) -> S3Code:
    if False:
        print('Hello World!')
    '\n    Stores the given lambda archive in an internal s3 bucket.\n    Also checks if zipfile matches the specifications\n\n    :param archive_file: Archive file to store\n    :param function_name: function name the archive should be stored for\n    :param region_name: region name the archive should be stored for\n    :param account_id: account id the archive should be stored for\n    :return: S3 Code object representing the archive stored in S3\n    '
    if not is_zip_file(archive_file):
        raise InvalidParameterValueException('Could not unzip uploaded file. Please check your file, then try to upload again.', Type='User')
    unzipped_size = get_unzipped_size(zip_file=io.BytesIO(archive_file))
    if unzipped_size >= config.LAMBDA_LIMITS_CODE_SIZE_UNZIPPED:
        raise InvalidParameterValueException(f'Unzipped size must be smaller than {config.LAMBDA_LIMITS_CODE_SIZE_UNZIPPED} bytes', Type='User')
    s3_client = connect_to(region_name=AWS_REGION_US_EAST_1, aws_access_key_id=config.INTERNAL_RESOURCE_ACCOUNT).s3
    bucket_name = f'awslambda-{region_name}-tasks'
    s3_client.create_bucket(Bucket=bucket_name)
    code_id = f'{function_name}-{uuid.uuid4()}'
    key = f'snapshots/{account_id}/{code_id}'
    s3_client.upload_fileobj(Fileobj=io.BytesIO(archive_file), Bucket=bucket_name, Key=key)
    code_sha256 = to_str(base64.b64encode(sha256(archive_file).digest()))
    return S3Code(id=code_id, account_id=account_id, s3_bucket=bucket_name, s3_key=key, s3_object_version=None, code_sha256=code_sha256, code_size=len(archive_file))

def create_hot_reloading_code(path: str) -> HotReloadingCode:
    if False:
        i = 10
        return i + 15
    if not PurePosixPath(path).is_absolute() and (not PureWindowsPath(path).is_absolute()):
        raise InvalidParameterValueException(f'When using hot reloading, the archive key has to be an absolute path! Your archive key: {path}')
    return HotReloadingCode(host_path=path)

def store_s3_bucket_archive(archive_bucket: str, archive_key: str, archive_version: Optional[str], function_name: str, region_name: str, account_id: str) -> ArchiveCode:
    if False:
        i = 10
        return i + 15
    '\n    Takes the lambda archive stored in the given bucket and stores it in an internal s3 bucket\n\n    :param archive_bucket: Bucket the archive is stored in\n    :param archive_key: Key the archive is stored under\n    :param archive_version: Version of the archive object in the bucket\n    :param function_name: function name the archive should be stored for\n    :param region_name: region name the archive should be stored for\n    :param account_id: account id the archive should be stored for\n    :return: S3 Code object representing the archive stored in S3\n    '
    if archive_bucket == config.BUCKET_MARKER_LOCAL:
        usage.hotreload.increment()
        return create_hot_reloading_code(path=archive_key)
    s3_client: 'S3Client' = connect_to().s3
    kwargs = {'VersionId': archive_version} if archive_version else {}
    archive_file = s3_client.get_object(Bucket=archive_bucket, Key=archive_key, **kwargs)['Body'].read()
    return store_lambda_archive(archive_file, function_name=function_name, region_name=region_name, account_id=account_id)

def create_image_code(image_uri: str) -> ImageCode:
    if False:
        return 10
    '\n    Creates an image code by inspecting the provided image\n\n    :param image_uri: Image URI of the image to inspect\n    :return: Image code object\n    '
    code_sha256 = '<cannot-find-image>'
    try:
        CONTAINER_CLIENT.pull_image(docker_image=image_uri)
    except ContainerException:
        LOG.debug('Cannot pull image %s. Maybe only available locally?', image_uri)
    try:
        code_sha256 = CONTAINER_CLIENT.inspect_image(image_name=image_uri)['RepoDigests'][0].rpartition(':')[2]
    except Exception as e:
        LOG.debug('Cannot inspect image %s. Is this image and/or docker available: %s', image_uri, e)
    return ImageCode(image_uri=image_uri, code_sha256=code_sha256, repository_type='ECR')