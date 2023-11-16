"""
File keeps Factory method to prepare required puller information
with its producers and consumers
"""
import logging
from typing import List, Optional
from botocore.exceptions import ClientError
from samcli.commands.exceptions import UserException
from samcli.commands.logs.console_consumers import CWConsoleEventConsumer
from samcli.commands.traces.traces_puller_factory import generate_trace_puller
from samcli.lib.observability.cw_logs.cw_log_formatters import CWAddNewLineIfItDoesntExist, CWColorizeErrorsFormatter, CWJsonFormatter, CWKeywordHighlighterFormatter, CWLogEventJSONMapper, CWPrettyPrintFormatter
from samcli.lib.observability.cw_logs.cw_log_group_provider import LogGroupProvider
from samcli.lib.observability.cw_logs.cw_log_puller import CWLogPuller
from samcli.lib.observability.observability_info_puller import ObservabilityCombinedPuller, ObservabilityEventConsumer, ObservabilityEventConsumerDecorator, ObservabilityPuller
from samcli.lib.observability.util import OutputOption
from samcli.lib.utils.boto_utils import BotoProviderType, get_client_error_code
from samcli.lib.utils.cloudformation import CloudFormationResourceSummary
from samcli.lib.utils.colors import Colored
LOG = logging.getLogger(__name__)

class NoPullerGeneratedException(UserException):
    """
    Used to indicate that no puller information have been generated
    therefore there is no observability information (logs, xray) to pull
    """

def generate_puller(boto_client_provider: BotoProviderType, resource_information_list: List[CloudFormationResourceSummary], filter_pattern: Optional[str]=None, additional_cw_log_groups: Optional[List[str]]=None, output: OutputOption=OutputOption.text, include_tracing: bool=False) -> ObservabilityPuller:
    if False:
        for i in range(10):
            print('nop')
    '\n    This function will generate generic puller which can be used to\n    pull information from various observability resources.\n\n    Parameters\n    ----------\n    boto_client_provider: BotoProviderType\n        Boto3 client generator, which will create a new instance of the client with a new session that could be\n        used within different threads/coroutines\n    resource_information_list : List[CloudFormationResourceSummary]\n        List of resource information, which keeps logical id, physical id and type of the resources\n    filter_pattern : Optional[str]\n        Optional filter pattern which will be used to filter incoming events\n    additional_cw_log_groups : Optional[str]\n        Optional list of additional CloudWatch log groups which will be used to fetch\n        log events from.\n    output : OutputOption\n        Decides how the output will be presented in the console. It is been used to select correct consumer type\n        between (default) text consumer or json consumer\n    include_tracing: bool\n        A flag to include the xray traces log or not\n\n    Returns\n    -------\n        Puller instance that can be used to pull information.\n    '
    if additional_cw_log_groups is None:
        additional_cw_log_groups = []
    pullers: List[ObservabilityPuller] = []
    for resource_information in resource_information_list:
        cw_log_group_name = LogGroupProvider.for_resource(boto_client_provider, resource_information.resource_type, resource_information.physical_resource_id)
        if not cw_log_group_name:
            LOG.debug("Can't find CloudWatch LogGroup name for resource (%s)", resource_information.logical_resource_id)
            continue
        consumer = generate_consumer(filter_pattern, output, resource_information.logical_resource_id)
        pullers.append(CWLogPuller(boto_client_provider('logs'), consumer, cw_log_group_name, resource_information.logical_resource_id))
    for cw_log_group in additional_cw_log_groups:
        consumer = generate_consumer(filter_pattern, output)
        logs_client = boto_client_provider('logs')
        _validate_cw_log_group_name(cw_log_group, logs_client)
        pullers.append(CWLogPuller(logs_client, consumer, cw_log_group))
    if include_tracing:
        trace_puller = generate_trace_puller(boto_client_provider('xray'), output)
        pullers.append(trace_puller)
    if not pullers:
        raise NoPullerGeneratedException('No valid resources find to pull information')
    return ObservabilityCombinedPuller(pullers)

def _validate_cw_log_group_name(cw_log_group, logs_client):
    if False:
        i = 10
        return i + 15
    try:
        _ = logs_client.describe_log_streams(logGroupName=cw_log_group, limit=1)
    except ClientError as ex:
        if get_client_error_code(ex) == 'ResourceNotFoundException':
            LOG.warning('CloudWatch log group name (%s) does not exist.', cw_log_group)

def generate_consumer(filter_pattern: Optional[str]=None, output: OutputOption=OutputOption.text, resource_name: Optional[str]=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates consumer instance with the given variables.\n    If output is JSON, then it will return consumer with formatters for just JSON.\n    Otherwise, it will return regular text console consumer\n    '
    if output == OutputOption.json:
        return generate_json_consumer()
    return generate_text_consumer(filter_pattern)

def generate_json_consumer() -> ObservabilityEventConsumer:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates event consumer, which prints CW Log Events as JSON into terminal\n\n    Returns\n    -------\n        ObservabilityEventConsumer which will store events into a file\n    '
    return ObservabilityEventConsumerDecorator([CWLogEventJSONMapper()], CWConsoleEventConsumer(True))

def generate_text_consumer(filter_pattern: Optional[str]) -> ObservabilityEventConsumer:
    if False:
        i = 10
        return i + 15
    "\n    Creates a console event consumer, which is used to display events in the user's console\n\n    Parameters\n    ----------\n    filter_pattern : str\n        Filter pattern is used to display certain words in a different pattern then\n        the rest of the messages.\n\n    Returns\n    -------\n        A consumer which will display events into console\n    "
    colored = Colored()
    return ObservabilityEventConsumerDecorator([CWColorizeErrorsFormatter(colored), CWJsonFormatter(), CWKeywordHighlighterFormatter(colored, filter_pattern), CWPrettyPrintFormatter(colored), CWAddNewLineIfItDoesntExist()], CWConsoleEventConsumer())