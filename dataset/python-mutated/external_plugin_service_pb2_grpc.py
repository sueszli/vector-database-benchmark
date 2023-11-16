"""Client and server classes corresponding to protobuf-defined services."""
import grpc
from flyteidl.service import external_plugin_service_pb2 as flyteidl_dot_service_dot_external__plugin__service__pb2

class ExternalPluginServiceStub(object):
    """ExternalPluginService defines an RPC Service that allows propeller to send the request to the backend plugin server.
    """

    def __init__(self, channel):
        if False:
            print('Hello World!')
        'Constructor.\n\n        Args:\n            channel: A grpc.Channel.\n        '
        self.CreateTask = channel.unary_unary('/flyteidl.service.ExternalPluginService/CreateTask', request_serializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskCreateRequest.SerializeToString, response_deserializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskCreateResponse.FromString)
        self.GetTask = channel.unary_unary('/flyteidl.service.ExternalPluginService/GetTask', request_serializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskGetRequest.SerializeToString, response_deserializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskGetResponse.FromString)
        self.DeleteTask = channel.unary_unary('/flyteidl.service.ExternalPluginService/DeleteTask', request_serializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskDeleteRequest.SerializeToString, response_deserializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskDeleteResponse.FromString)

class ExternalPluginServiceServicer(object):
    """ExternalPluginService defines an RPC Service that allows propeller to send the request to the backend plugin server.
    """

    def CreateTask(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        'Send a task create request to the backend plugin server.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTask(self, request, context):
        if False:
            print('Hello World!')
        'Get job status.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteTask(self, request, context):
        if False:
            i = 10
            return i + 15
        'Delete the task resource.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_ExternalPluginServiceServicer_to_server(servicer, server):
    if False:
        return 10
    rpc_method_handlers = {'CreateTask': grpc.unary_unary_rpc_method_handler(servicer.CreateTask, request_deserializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskCreateRequest.FromString, response_serializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskCreateResponse.SerializeToString), 'GetTask': grpc.unary_unary_rpc_method_handler(servicer.GetTask, request_deserializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskGetRequest.FromString, response_serializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskGetResponse.SerializeToString), 'DeleteTask': grpc.unary_unary_rpc_method_handler(servicer.DeleteTask, request_deserializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskDeleteRequest.FromString, response_serializer=flyteidl_dot_service_dot_external__plugin__service__pb2.TaskDeleteResponse.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('flyteidl.service.ExternalPluginService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

class ExternalPluginService(object):
    """ExternalPluginService defines an RPC Service that allows propeller to send the request to the backend plugin server.
    """

    @staticmethod
    def CreateTask(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            print('Hello World!')
        return grpc.experimental.unary_unary(request, target, '/flyteidl.service.ExternalPluginService/CreateTask', flyteidl_dot_service_dot_external__plugin__service__pb2.TaskCreateRequest.SerializeToString, flyteidl_dot_service_dot_external__plugin__service__pb2.TaskCreateResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTask(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        return grpc.experimental.unary_unary(request, target, '/flyteidl.service.ExternalPluginService/GetTask', flyteidl_dot_service_dot_external__plugin__service__pb2.TaskGetRequest.SerializeToString, flyteidl_dot_service_dot_external__plugin__service__pb2.TaskGetResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteTask(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            i = 10
            return i + 15
        return grpc.experimental.unary_unary(request, target, '/flyteidl.service.ExternalPluginService/DeleteTask', flyteidl_dot_service_dot_external__plugin__service__pb2.TaskDeleteRequest.SerializeToString, flyteidl_dot_service_dot_external__plugin__service__pb2.TaskDeleteResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)