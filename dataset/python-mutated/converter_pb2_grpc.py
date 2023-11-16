"""Client and server classes corresponding to protobuf-defined services."""
import grpc
from . import converter_pb2 as pulumi_dot_converter__pb2

class ConverterStub(object):
    """Converter is a service for converting between other ecosystems and Pulumi.
    This is currently unstable and experimental.
    """

    def __init__(self, channel):
        if False:
            print('Hello World!')
        'Constructor.\n\n        Args:\n            channel: A grpc.Channel.\n        '
        self.ConvertState = channel.unary_unary('/pulumirpc.Converter/ConvertState', request_serializer=pulumi_dot_converter__pb2.ConvertStateRequest.SerializeToString, response_deserializer=pulumi_dot_converter__pb2.ConvertStateResponse.FromString)
        self.ConvertProgram = channel.unary_unary('/pulumirpc.Converter/ConvertProgram', request_serializer=pulumi_dot_converter__pb2.ConvertProgramRequest.SerializeToString, response_deserializer=pulumi_dot_converter__pb2.ConvertProgramResponse.FromString)

class ConverterServicer(object):
    """Converter is a service for converting between other ecosystems and Pulumi.
    This is currently unstable and experimental.
    """

    def ConvertState(self, request, context):
        if False:
            print('Hello World!')
        'ConvertState converts state from the target ecosystem into a form that can be imported into Pulumi.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ConvertProgram(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        'ConvertProgram converts a program from the target ecosystem into a form that can be used with Pulumi.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_ConverterServicer_to_server(servicer, server):
    if False:
        return 10
    rpc_method_handlers = {'ConvertState': grpc.unary_unary_rpc_method_handler(servicer.ConvertState, request_deserializer=pulumi_dot_converter__pb2.ConvertStateRequest.FromString, response_serializer=pulumi_dot_converter__pb2.ConvertStateResponse.SerializeToString), 'ConvertProgram': grpc.unary_unary_rpc_method_handler(servicer.ConvertProgram, request_deserializer=pulumi_dot_converter__pb2.ConvertProgramRequest.FromString, response_serializer=pulumi_dot_converter__pb2.ConvertProgramResponse.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('pulumirpc.Converter', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

class Converter(object):
    """Converter is a service for converting between other ecosystems and Pulumi.
    This is currently unstable and experimental.
    """

    @staticmethod
    def ConvertState(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            print('Hello World!')
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.Converter/ConvertState', pulumi_dot_converter__pb2.ConvertStateRequest.SerializeToString, pulumi_dot_converter__pb2.ConvertStateResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ConvertProgram(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            print('Hello World!')
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.Converter/ConvertProgram', pulumi_dot_converter__pb2.ConvertProgramRequest.SerializeToString, pulumi_dot_converter__pb2.ConvertProgramResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)