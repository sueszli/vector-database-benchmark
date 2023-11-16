"""
Examples to show usage of the azure-core-tracing-opentelemetry
with the Eventhub SDK.

This example traces calls for sending a batch to eventhub.

An alternative path to export using the OpenTelemetry exporter for Azure Monitor
is also mentioned in the sample. Please take a look at the commented code.
"""
from azure.core.settings import settings
settings.tracing_implementation = 'opentelemetry'
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
exporter = ConsoleSpanExporter()
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
from azure.eventhub import EventHubProducerClient, EventData, EventHubConsumerClient
import os
FULLY_QUALIFIED_NAMESPACE = os.environ['EVENT_HUB_HOSTNAME']
EVENTHUB_NAME = os.environ['EVENT_HUB_NAME']
credential = os.environ['EVENTHUB_CONN_STR']

def on_event(partition_context, event):
    if False:
        while True:
            i = 10
    print('Received event from partition: {}.'.format(partition_context.partition_id))

def on_partition_initialize(partition_context):
    if False:
        for i in range(10):
            print('nop')
    print('Partition: {} has been initialized.'.format(partition_context.partition_id))

def on_partition_close(partition_context, reason):
    if False:
        i = 10
        return i + 15
    print('Partition: {} has been closed, reason for closing: {}.'.format(partition_context.partition_id, reason))

def on_error(partition_context, error):
    if False:
        print('Hello World!')
    if partition_context:
        print('An exception: {} occurred during receiving from Partition: {}.'.format(partition_context.partition_id, error))
    else:
        print('An exception: {} occurred during the load balance process.'.format(error))
with tracer.start_as_current_span(name='MyApplication'):
    consumer_client = EventHubConsumerClient.from_connection_string(conn_str=credential, consumer_group='$Default', eventhub_name=EVENTHUB_NAME)
    try:
        with consumer_client:
            consumer_client.receive(on_event=on_event, on_partition_initialize=on_partition_initialize, on_partition_close=on_partition_close, on_error=on_error, starting_position='-1')
    except KeyboardInterrupt:
        print('Stopped receiving.')