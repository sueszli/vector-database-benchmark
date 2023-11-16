import grpc
from tensorflow.core.debug import debug_service_pb2 as tensorflow_dot_core_dot_debug_dot_debug__service__pb2
from tensorflow.core.protobuf import debug_pb2 as tensorflow_dot_core_dot_protobuf_dot_debug__pb2
from tensorflow.core.util import event_pb2 as tensorflow_dot_core_dot_util_dot_event__pb2

class EventListenerStub(object):
    """EventListener: Receives Event protos, e.g., from debugged TensorFlow
  runtime(s).
  """

    def __init__(self, channel):
        if False:
            while True:
                i = 10
        'Constructor.\n\n    Args:\n      channel: A grpc.Channel.\n    '
        self.SendEvents = channel.stream_stream('/tensorflow.EventListener/SendEvents', request_serializer=tensorflow_dot_core_dot_util_dot_event__pb2.Event.SerializeToString, response_deserializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.EventReply.FromString)
        self.SendTracebacks = channel.unary_unary('/tensorflow.EventListener/SendTracebacks', request_serializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.CallTraceback.SerializeToString, response_deserializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.EventReply.FromString)
        self.SendSourceFiles = channel.unary_unary('/tensorflow.EventListener/SendSourceFiles', request_serializer=tensorflow_dot_core_dot_protobuf_dot_debug__pb2.DebuggedSourceFiles.SerializeToString, response_deserializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.EventReply.FromString)

class EventListenerServicer(object):
    """EventListener: Receives Event protos, e.g., from debugged TensorFlow
  runtime(s).
  """

    def SendEvents(self, request_iterator, context):
        if False:
            print('Hello World!')
        'Client(s) can use this RPC method to send the EventListener Event protos.\n    The Event protos can hold information such as:\n    1) intermediate tensors from a debugged graph being executed, which can\n    be sent from DebugIdentity ops configured with grpc URLs.\n    2) GraphDefs of partition graphs, which can be sent from special debug\n    ops that get executed immediately after the beginning of the graph\n    execution.\n    '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendTracebacks(self, request, context):
        if False:
            while True:
                i = 10
        'Send the tracebacks of ops in a Python graph definition.\n    '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendSourceFiles(self, request, context):
        if False:
            i = 10
            return i + 15
        'Send a collection of source code files being debugged.\n    '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_EventListenerServicer_to_server(servicer, server):
    if False:
        print('Hello World!')
    rpc_method_handlers = {'SendEvents': grpc.stream_stream_rpc_method_handler(servicer.SendEvents, request_deserializer=tensorflow_dot_core_dot_util_dot_event__pb2.Event.FromString, response_serializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.EventReply.SerializeToString), 'SendTracebacks': grpc.unary_unary_rpc_method_handler(servicer.SendTracebacks, request_deserializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.CallTraceback.FromString, response_serializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.EventReply.SerializeToString), 'SendSourceFiles': grpc.unary_unary_rpc_method_handler(servicer.SendSourceFiles, request_deserializer=tensorflow_dot_core_dot_protobuf_dot_debug__pb2.DebuggedSourceFiles.FromString, response_serializer=tensorflow_dot_core_dot_debug_dot_debug__service__pb2.EventReply.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('tensorflow.EventListener', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))