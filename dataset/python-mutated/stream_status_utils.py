from datetime import datetime
from airbyte_cdk.models import AirbyteMessage, AirbyteStream, AirbyteStreamStatus, AirbyteStreamStatusTraceMessage, AirbyteTraceMessage, StreamDescriptor, TraceType
from airbyte_cdk.models import Type as MessageType

def as_airbyte_message(stream: AirbyteStream, current_status: AirbyteStreamStatus) -> AirbyteMessage:
    if False:
        while True:
            i = 10
    '\n    Builds an AirbyteStreamStatusTraceMessage for the provided stream\n    '
    now_millis = datetime.now().timestamp() * 1000.0
    trace_message = AirbyteTraceMessage(type=TraceType.STREAM_STATUS, emitted_at=now_millis, stream_status=AirbyteStreamStatusTraceMessage(stream_descriptor=StreamDescriptor(name=stream.name, namespace=stream.namespace), status=current_status))
    return AirbyteMessage(type=MessageType.TRACE, trace=trace_message)