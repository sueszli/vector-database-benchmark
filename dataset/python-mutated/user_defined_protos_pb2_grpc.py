"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import user_defined_protos_pb2 as user__defined__protos__pb2

class UserDefinedServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.\n\n        Args:\n            channel: A grpc.Channel.\n        '
        self.__call__ = channel.unary_unary('/userdefinedprotos.UserDefinedService/__call__', request_serializer=user__defined__protos__pb2.UserDefinedMessage.SerializeToString, response_deserializer=user__defined__protos__pb2.UserDefinedResponse.FromString)
        self.Multiplexing = channel.unary_unary('/userdefinedprotos.UserDefinedService/Multiplexing', request_serializer=user__defined__protos__pb2.UserDefinedMessage2.SerializeToString, response_deserializer=user__defined__protos__pb2.UserDefinedResponse2.FromString)
        self.Streaming = channel.unary_stream('/userdefinedprotos.UserDefinedService/Streaming', request_serializer=user__defined__protos__pb2.UserDefinedMessage.SerializeToString, response_deserializer=user__defined__protos__pb2.UserDefinedResponse.FromString)

class UserDefinedServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def __call__(self, request, context):
        if False:
            while True:
                i = 10
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Multiplexing(self, request, context):
        if False:
            i = 10
            return i + 15
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Streaming(self, request, context):
        if False:
            while True:
                i = 10
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_UserDefinedServiceServicer_to_server(servicer, server):
    if False:
        print('Hello World!')
    rpc_method_handlers = {'__call__': grpc.unary_unary_rpc_method_handler(servicer.__call__, request_deserializer=user__defined__protos__pb2.UserDefinedMessage.FromString, response_serializer=user__defined__protos__pb2.UserDefinedResponse.SerializeToString), 'Multiplexing': grpc.unary_unary_rpc_method_handler(servicer.Multiplexing, request_deserializer=user__defined__protos__pb2.UserDefinedMessage2.FromString, response_serializer=user__defined__protos__pb2.UserDefinedResponse2.SerializeToString), 'Streaming': grpc.unary_stream_rpc_method_handler(servicer.Streaming, request_deserializer=user__defined__protos__pb2.UserDefinedMessage.FromString, response_serializer=user__defined__protos__pb2.UserDefinedResponse.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('userdefinedprotos.UserDefinedService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

class UserDefinedService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def __call__(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            while True:
                i = 10
        return grpc.experimental.unary_unary(request, target, '/userdefinedprotos.UserDefinedService/__call__', user__defined__protos__pb2.UserDefinedMessage.SerializeToString, user__defined__protos__pb2.UserDefinedResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Multiplexing(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        return grpc.experimental.unary_unary(request, target, '/userdefinedprotos.UserDefinedService/Multiplexing', user__defined__protos__pb2.UserDefinedMessage2.SerializeToString, user__defined__protos__pb2.UserDefinedResponse2.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Streaming(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            return 10
        return grpc.experimental.unary_stream(request, target, '/userdefinedprotos.UserDefinedService/Streaming', user__defined__protos__pb2.UserDefinedMessage.SerializeToString, user__defined__protos__pb2.UserDefinedResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

class ImageClassificationServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        if False:
            while True:
                i = 10
        'Constructor.\n\n        Args:\n            channel: A grpc.Channel.\n        '
        self.Predict = channel.unary_unary('/userdefinedprotos.ImageClassificationService/Predict', request_serializer=user__defined__protos__pb2.ImageData.SerializeToString, response_deserializer=user__defined__protos__pb2.ImageClass.FromString)

class ImageClassificationServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Predict(self, request, context):
        if False:
            return 10
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_ImageClassificationServiceServicer_to_server(servicer, server):
    if False:
        print('Hello World!')
    rpc_method_handlers = {'Predict': grpc.unary_unary_rpc_method_handler(servicer.Predict, request_deserializer=user__defined__protos__pb2.ImageData.FromString, response_serializer=user__defined__protos__pb2.ImageClass.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('userdefinedprotos.ImageClassificationService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

class ImageClassificationService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Predict(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            return 10
        return grpc.experimental.unary_unary(request, target, '/userdefinedprotos.ImageClassificationService/Predict', user__defined__protos__pb2.ImageData.SerializeToString, user__defined__protos__pb2.ImageClass.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)