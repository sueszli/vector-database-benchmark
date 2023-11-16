"""Client and server classes corresponding to protobuf-defined services."""
import grpc
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from . import engine_pb2 as pulumi_dot_engine__pb2

class EngineStub(object):
    """Engine is an auxiliary service offered to language and resource provider plugins. Its main purpose today is
    to serve as a common logging endpoint, but it also serves as a state storage mechanism for language hosts
    that can't store their own global state.
    """

    def __init__(self, channel):
        if False:
            while True:
                i = 10
        'Constructor.\n\n        Args:\n            channel: A grpc.Channel.\n        '
        self.Log = channel.unary_unary('/pulumirpc.Engine/Log', request_serializer=pulumi_dot_engine__pb2.LogRequest.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.GetRootResource = channel.unary_unary('/pulumirpc.Engine/GetRootResource', request_serializer=pulumi_dot_engine__pb2.GetRootResourceRequest.SerializeToString, response_deserializer=pulumi_dot_engine__pb2.GetRootResourceResponse.FromString)
        self.SetRootResource = channel.unary_unary('/pulumirpc.Engine/SetRootResource', request_serializer=pulumi_dot_engine__pb2.SetRootResourceRequest.SerializeToString, response_deserializer=pulumi_dot_engine__pb2.SetRootResourceResponse.FromString)

class EngineServicer(object):
    """Engine is an auxiliary service offered to language and resource provider plugins. Its main purpose today is
    to serve as a common logging endpoint, but it also serves as a state storage mechanism for language hosts
    that can't store their own global state.
    """

    def Log(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        'Log logs a global message in the engine, including errors and warnings.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetRootResource(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        'GetRootResource gets the URN of the root resource, the resource that should be the root of all\n        otherwise-unparented resources.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetRootResource(self, request, context):
        if False:
            while True:
                i = 10
        'SetRootResource sets the URN of the root resource.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_EngineServicer_to_server(servicer, server):
    if False:
        for i in range(10):
            print('nop')
    rpc_method_handlers = {'Log': grpc.unary_unary_rpc_method_handler(servicer.Log, request_deserializer=pulumi_dot_engine__pb2.LogRequest.FromString, response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString), 'GetRootResource': grpc.unary_unary_rpc_method_handler(servicer.GetRootResource, request_deserializer=pulumi_dot_engine__pb2.GetRootResourceRequest.FromString, response_serializer=pulumi_dot_engine__pb2.GetRootResourceResponse.SerializeToString), 'SetRootResource': grpc.unary_unary_rpc_method_handler(servicer.SetRootResource, request_deserializer=pulumi_dot_engine__pb2.SetRootResourceRequest.FromString, response_serializer=pulumi_dot_engine__pb2.SetRootResourceResponse.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('pulumirpc.Engine', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

class Engine(object):
    """Engine is an auxiliary service offered to language and resource provider plugins. Its main purpose today is
    to serve as a common logging endpoint, but it also serves as a state storage mechanism for language hosts
    that can't store their own global state.
    """

    @staticmethod
    def Log(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            print('Hello World!')
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.Engine/Log', pulumi_dot_engine__pb2.LogRequest.SerializeToString, google_dot_protobuf_dot_empty__pb2.Empty.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetRootResource(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            return 10
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.Engine/GetRootResource', pulumi_dot_engine__pb2.GetRootResourceRequest.SerializeToString, pulumi_dot_engine__pb2.GetRootResourceResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetRootResource(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            while True:
                i = 10
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.Engine/SetRootResource', pulumi_dot_engine__pb2.SetRootResourceRequest.SerializeToString, pulumi_dot_engine__pb2.SetRootResourceResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)