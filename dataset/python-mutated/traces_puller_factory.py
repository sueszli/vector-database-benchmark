"""
Factory methods which generates puller and consumer instances for XRay events
"""
from typing import Any, List
from samcli.commands.traces.trace_console_consumers import XRayTraceConsoleConsumer
from samcli.lib.observability.observability_info_puller import ObservabilityCombinedPuller, ObservabilityEventConsumer, ObservabilityEventConsumerDecorator, ObservabilityPuller
from samcli.lib.observability.util import OutputOption
from samcli.lib.observability.xray_traces.xray_event_mappers import XRayServiceGraphConsoleMapper, XRayServiceGraphJSONMapper, XRayTraceConsoleMapper, XRayTraceJSONMapper
from samcli.lib.observability.xray_traces.xray_event_puller import XRayTracePuller
from samcli.lib.observability.xray_traces.xray_service_graph_event_puller import XRayServiceGraphPuller

def generate_trace_puller(xray_client: Any, output: OutputOption=OutputOption.text) -> ObservabilityPuller:
    if False:
        while True:
            i = 10
    '\n    Generates puller instance with correct consumer and/or mapper configuration\n\n    Parameters\n    ----------\n    xray_client : Any\n        boto3 xray client to be used in XRayTracePuller instance\n    output : OutputOption\n        Decides how the output will be presented in the console. It is been used to select correct consumer type\n        between (default) text consumer or json consumer\n\n    Returns\n    -------\n        Puller instance with desired configuration\n    '
    pullers: List[ObservabilityPuller] = []
    pullers.append(XRayTracePuller(xray_client, generate_xray_event_consumer(output)))
    pullers.append(XRayServiceGraphPuller(xray_client, generate_xray_service_graph_consumer(output)))
    return ObservabilityCombinedPuller(pullers)

def generate_json_xray_event_consumer() -> ObservabilityEventConsumer:
    if False:
        i = 10
        return i + 15
    '\n    Generates unformatted consumer, which will print XRay events unformatted JSON into terminal\n\n    Returns\n    -------\n        File consumer instance with desired mapper configuration\n    '
    return ObservabilityEventConsumerDecorator([XRayTraceJSONMapper()], XRayTraceConsoleConsumer())

def generate_xray_event_console_consumer() -> ObservabilityEventConsumer:
    if False:
        i = 10
        return i + 15
    '\n    Generates an instance of event consumer which will print events into console\n\n    Returns\n    -------\n        Console consumer instance with desired mapper configuration\n    '
    return ObservabilityEventConsumerDecorator([XRayTraceConsoleMapper()], XRayTraceConsoleConsumer())

def generate_xray_event_consumer(output: OutputOption=OutputOption.text) -> ObservabilityEventConsumer:
    if False:
        return 10
    '\n    Generates consumer instance with the given variables.\n    If output is JSON, then it will return consumer with formatters for just JSON.\n    Otherwise, it will return regular text console consumer\n    '
    if output == OutputOption.json:
        return generate_json_xray_event_consumer()
    return generate_xray_event_console_consumer()

def generate_json_xray_service_graph_consumer() -> ObservabilityEventConsumer:
    if False:
        print('Hello World!')
    '\n    Generates unformatted consumer, which will print XRay events unformatted JSON into terminal\n\n    Returns\n    -------\n        File consumer instance with desired mapper configuration\n    '
    return ObservabilityEventConsumerDecorator([XRayServiceGraphJSONMapper()], XRayTraceConsoleConsumer())

def generate_xray_service_graph_console_consumer() -> ObservabilityEventConsumer:
    if False:
        while True:
            i = 10
    '\n    Generates an instance of event consumer which will print events into console\n\n    Returns\n    -------\n        Console consumer instance with desired mapper configuration\n    '
    return ObservabilityEventConsumerDecorator([XRayServiceGraphConsoleMapper()], XRayTraceConsoleConsumer())

def generate_xray_service_graph_consumer(output: OutputOption=OutputOption.text) -> ObservabilityEventConsumer:
    if False:
        return 10
    '\n    Generates consumer instance with the given variables.\n    If output is JSON, then it will return consumer with formatters for just JSON.\n    Otherwise, it will return regular text console consumer\n    '
    if output == OutputOption.json:
        return generate_json_xray_service_graph_consumer()
    return generate_xray_service_graph_console_consumer()