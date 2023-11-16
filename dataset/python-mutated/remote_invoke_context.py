"""
Context object used by `sam remote invoke` command
"""
import logging
from dataclasses import dataclass
from typing import Optional, cast
from botocore.exceptions import ClientError
from samcli.commands.remote.exceptions import AmbiguousResourceForRemoteInvoke, InvalidRemoteInvokeParameters, InvalidStackNameProvidedForRemoteInvoke, NoResourceFoundForRemoteInvoke, ResourceNotSupportedForRemoteInvoke, UnsupportedServiceForRemoteInvoke
from samcli.lib.remote_invoke.exceptions import ErrorBotoApiCallException
from samcli.lib.remote_invoke.remote_invoke_executor_factory import RemoteInvokeExecutorFactory
from samcli.lib.remote_invoke.remote_invoke_executors import RemoteInvokeConsumer, RemoteInvokeExecutionInfo, RemoteInvokeLogOutput, RemoteInvokeResponse
from samcli.lib.remote_invoke.sqs_invoke_executors import get_queue_url_from_arn
from samcli.lib.schemas.schemas_api_caller import SchemasApiCaller
from samcli.lib.shared_test_events.lambda_shared_test_event import LambdaSharedTestEvent
from samcli.lib.utils import osutils
from samcli.lib.utils.arn_utils import ARNParts, InvalidArnValue
from samcli.lib.utils.boto_utils import BotoProviderType, get_client_error_code
from samcli.lib.utils.cloudformation import CloudFormationResourceSummary, get_resource_summaries, get_resource_summary, get_resource_summary_from_physical_id
from samcli.lib.utils.resources import AWS_KINESIS_STREAM, AWS_LAMBDA_FUNCTION, AWS_SQS_QUEUE, AWS_STEPFUNCTIONS_STATEMACHINE
from samcli.lib.utils.stream_writer import StreamWriter
LOG = logging.getLogger(__name__)
SUPPORTED_SERVICES = {'lambda': AWS_LAMBDA_FUNCTION, 'states': AWS_STEPFUNCTIONS_STATEMACHINE, 'sqs': AWS_SQS_QUEUE, 'kinesis': AWS_KINESIS_STREAM}
RESOURCES_PRIORITY_ORDER = [AWS_LAMBDA_FUNCTION, AWS_STEPFUNCTIONS_STATEMACHINE, AWS_SQS_QUEUE, AWS_KINESIS_STREAM]

class RemoteInvokeContext:
    _boto_client_provider: BotoProviderType
    _boto_resource_provider: BotoProviderType
    _stack_name: Optional[str]
    _resource_id: Optional[str]
    _resource_summary: Optional[CloudFormationResourceSummary]

    def __init__(self, boto_client_provider: BotoProviderType, boto_resource_provider: BotoProviderType, stack_name: Optional[str], resource_id: Optional[str]):
        if False:
            print('Hello World!')
        self._boto_resource_provider = boto_resource_provider
        self._boto_client_provider = boto_client_provider
        self._stack_name = stack_name
        self._resource_id = resource_id
        self._resource_summary = None

    def __enter__(self) -> 'RemoteInvokeContext':
        if False:
            for i in range(10):
                print('nop')
        self._populate_resource_summary()
        return self

    def __exit__(self, *args) -> None:
        if False:
            return 10
        pass

    def run(self, remote_invoke_input: RemoteInvokeExecutionInfo) -> None:
        if False:
            while True:
                i = 10
        '\n        Instantiates remote invoke executor with populated resource summary information, executes it with the provided\n        input & returns its response back to the caller. If resource is not supported by command, raises\n        ResourceNotSupportedForRemoteInvoke exception.\n\n        Parameters\n        ----------\n        remote_invoke_input: RemoteInvokeExecutionInfo\n            RemoteInvokeExecutionInfo which contains the payload and other information that will be required during\n            the invocation\n        '
        if not self.resource_summary:
            raise self.missing_resource_exception()
        remote_invoke_executor_factory = RemoteInvokeExecutorFactory(self._boto_client_provider)
        remote_invoke_executor = remote_invoke_executor_factory.create_remote_invoke_executor(self.resource_summary, remote_invoke_input.output_format, DefaultRemoteInvokeResponseConsumer(self.stdout), DefaultRemoteInvokeLogConsumer(self.stderr))
        if not remote_invoke_executor:
            raise ResourceNotSupportedForRemoteInvoke(f'Resource type {self.resource_summary.resource_type} is not supported for remote invoke.')
        remote_invoke_executor.execute(remote_invoke_input)

    @property
    def resource_summary(self):
        if False:
            i = 10
            return i + 15
        if not self._resource_summary:
            self._populate_resource_summary()
        return self._resource_summary

    def missing_resource_exception(self):
        if False:
            for i in range(10):
                print('nop')
        return AmbiguousResourceForRemoteInvoke(f"Can't find resource information from stack name ({self._stack_name}) and resource id ({self._resource_id})")

    def get_lambda_shared_test_event_provider(self):
        if False:
            while True:
                i = 10
        schemas_client = self._boto_client_provider('schemas')
        lambda_client = self._boto_client_provider('lambda')
        api_caller = SchemasApiCaller(schemas_client)
        lambda_test_event = LambdaSharedTestEvent(api_caller, lambda_client)
        return lambda_test_event

    def _populate_resource_summary(self) -> None:
        if False:
            return 10
        '\n        Populates self._resource_summary field from self._stack_name and/or self._resource_id\n\n        Either self._stack_name or self._resource_id should be defined, it fails otherwise.\n\n        If only self._stack_name is defined, it tries to find single resource in that stack,\n        see _get_single_resource_from_stack for details.\n\n        If only self._resource_id is defined, it tries to parse its ARN or validate it as physical id,\n        see _get_from_physical_resource_id for details.\n        '
        if not self._stack_name and (not self._resource_id):
            raise InvalidRemoteInvokeParameters('Either --stack-name option or resource_id argument should be provided')
        try:
            if not self._resource_id:
                self._resource_summary = self._get_single_resource_from_stack()
                self._resource_id = self._resource_summary.logical_resource_id
                return
            if not self._stack_name:
                self._resource_summary = self._get_from_physical_resource_id()
                return
            self._resource_summary = get_resource_summary(self._boto_resource_provider, self._boto_client_provider, self._stack_name, self._resource_id)
        except ClientError as ex:
            error_code = get_client_error_code(ex)
            if error_code == 'ValidationError':
                raise InvalidStackNameProvidedForRemoteInvoke(f"Invalid --stack-name parameter. Stack with id '{self._stack_name}' does not exist")
            raise ex

    def _get_single_resource_from_stack(self) -> CloudFormationResourceSummary:
        if False:
            print('Hello World!')
        '\n        Queries all resources from stack with its type,\n        and returns its information if stack has only one resource from that type (including nested stacks)\n        '
        LOG.debug('Trying to get single resource with %s type in %s stack since no resource id is provided', RESOURCES_PRIORITY_ORDER, self._stack_name)
        resource_summaries = get_resource_summaries(self._boto_resource_provider, self._boto_client_provider, cast(str, self._stack_name), set(RESOURCES_PRIORITY_ORDER))
        if len(resource_summaries) == 1:
            ((logical_id, resource_summary),) = resource_summaries.items()
            LOG.debug('Using %s resource for remote invocation (%s)', logical_id, resource_summary)
            return resource_summary
        if len(resource_summaries) > 1:
            for resource_type in RESOURCES_PRIORITY_ORDER:
                resource_type_count = 0
                single_resource_summary = None
                for (logical_id, resource_summary) in resource_summaries.items():
                    if resource_summary.resource_type == resource_type:
                        resource_type_count += 1
                        single_resource_summary = resource_summary
                if resource_type_count == 1 and single_resource_summary:
                    return single_resource_summary
                elif resource_type_count > 1:
                    raise AmbiguousResourceForRemoteInvoke(f'{self._stack_name} contains more than one resource that could be used with remote invoke, please provide resource_id argument to resolve ambiguity.')
        raise NoResourceFoundForRemoteInvoke(f'{self._stack_name} stack has no resources that can be used with remote invoke.')

    def _get_from_physical_resource_id(self) -> CloudFormationResourceSummary:
        if False:
            return 10
        '\n        It first tries to parse given string as ARN and extracts the service name out of it. If it succeeds and that\n        service is supported, it generates CloudFormationResourceSummary out of that information\n\n        If it fails, it tries to resolve CloudFormationResourceSummary from the physical id of the resource\n        (see get_resource_summary_from_physical_id for details)\n        '
        resource_id = cast(str, self._resource_id)
        try:
            resource_arn = ARNParts(resource_id)
            service_from_arn = resource_arn.service
            if service_from_arn not in SUPPORTED_SERVICES:
                raise UnsupportedServiceForRemoteInvoke(f'{service_from_arn} is not supported service, please use an ARN for following services, {SUPPORTED_SERVICES}')
            if SUPPORTED_SERVICES.get(service_from_arn) == AWS_SQS_QUEUE:
                sqs_client = self._boto_client_provider('sqs')
                resource_id = get_queue_url_from_arn(sqs_client, resource_arn.resource_id)
            if SUPPORTED_SERVICES.get(service_from_arn) == AWS_KINESIS_STREAM:
                resource_id = resource_arn.resource_id
            return CloudFormationResourceSummary(cast(str, SUPPORTED_SERVICES.get(service_from_arn)), resource_id, resource_id)
        except InvalidArnValue:
            LOG.debug('Given %s is not an ARN, trying to get resource information from CloudFormation', self._resource_id)
            resource_summary = get_resource_summary_from_physical_id(self._boto_client_provider, resource_id)
            if not resource_summary:
                raise AmbiguousResourceForRemoteInvoke(f"Can't find exact resource information with given {self._resource_id}. Please provide full resource ARN or --stack-name to resolve the ambiguity.")
            return resource_summary
        except ErrorBotoApiCallException:
            raise AmbiguousResourceForRemoteInvoke(f"Can't find exact resource information with given {self._resource_id}. Please provide the correct ARN or --stack-name to resolve the ambiguity.")

    @property
    def stdout(self) -> StreamWriter:
        if False:
            i = 10
            return i + 15
        '\n        Returns stream writer for stdout to output Lambda function logs to\n\n        Returns\n        -------\n        samcli.lib.utils.stream_writer.StreamWriter\n            Stream writer for stdout\n        '
        stream = osutils.stdout()
        return StreamWriter(stream, auto_flush=True)

    @property
    def stderr(self) -> StreamWriter:
        if False:
            while True:
                i = 10
        '\n        Returns stream writer for stderr to output Lambda function errors to\n\n        Returns\n        -------\n        samcli.lib.utils.stream_writer.StreamWriter\n            Stream writer for stderr\n        '
        stream = osutils.stderr()
        return StreamWriter(stream, auto_flush=True)

@dataclass
class DefaultRemoteInvokeResponseConsumer(RemoteInvokeConsumer[RemoteInvokeResponse]):
    """
    Default RemoteInvokeResponse consumer, writes given response event to the configured StreamWriter
    """
    _stream_writer: StreamWriter

    def consume(self, remote_invoke_response: RemoteInvokeResponse) -> None:
        if False:
            while True:
                i = 10
        self._stream_writer.write_str(cast(str, remote_invoke_response.response))

@dataclass
class DefaultRemoteInvokeLogConsumer(RemoteInvokeConsumer[RemoteInvokeLogOutput]):
    """
    Default RemoteInvokeLogOutput consumer, writes given log event to the configured StreamWriter
    """
    _stream_writer: StreamWriter

    def consume(self, remote_invoke_response: RemoteInvokeLogOutput) -> None:
        if False:
            print('Hello World!')
        self._stream_writer.write_str(remote_invoke_response.log_output)