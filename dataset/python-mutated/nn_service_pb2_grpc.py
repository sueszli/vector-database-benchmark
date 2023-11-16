"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import nn_service_pb2 as nn__service__pb2

class NNServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        if False:
            return 10
        'Constructor.\n\n        Args:\n            channel: A grpc.Channel.\n        '
        self.train = channel.unary_unary('/nn.NNService/train', request_serializer=nn__service__pb2.TrainRequest.SerializeToString, response_deserializer=nn__service__pb2.TrainResponse.FromString)
        self.evaluate = channel.unary_unary('/nn.NNService/evaluate', request_serializer=nn__service__pb2.EvaluateRequest.SerializeToString, response_deserializer=nn__service__pb2.EvaluateResponse.FromString)
        self.predict = channel.unary_unary('/nn.NNService/predict', request_serializer=nn__service__pb2.PredictRequest.SerializeToString, response_deserializer=nn__service__pb2.PredictResponse.FromString)
        self.upload_meta = channel.unary_unary('/nn.NNService/upload_meta', request_serializer=nn__service__pb2.UploadMetaRequest.SerializeToString, response_deserializer=nn__service__pb2.UploadMetaResponse.FromString)
        self.upload_file = channel.stream_unary('/nn.NNService/upload_file', request_serializer=nn__service__pb2.ByteChunk.SerializeToString, response_deserializer=nn__service__pb2.UploadMetaResponse.FromString)
        self.save_server_model = channel.unary_unary('/nn.NNService/save_server_model', request_serializer=nn__service__pb2.SaveModelRequest.SerializeToString, response_deserializer=nn__service__pb2.SaveModelResponse.FromString)
        self.load_server_model = channel.unary_unary('/nn.NNService/load_server_model', request_serializer=nn__service__pb2.LoadModelRequest.SerializeToString, response_deserializer=nn__service__pb2.LoadModelResponse.FromString)

class NNServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def train(self, request, context):
        if False:
            i = 10
            return i + 15
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def evaluate(self, request, context):
        if False:
            return 10
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def predict(self, request, context):
        if False:
            while True:
                i = 10
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def upload_meta(self, request, context):
        if False:
            i = 10
            return i + 15
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def upload_file(self, request_iterator, context):
        if False:
            print('Hello World!')
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def save_server_model(self, request, context):
        if False:
            return 10
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def load_server_model(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        'Missing associated documentation comment in .proto file.'
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_NNServiceServicer_to_server(servicer, server):
    if False:
        for i in range(10):
            print('nop')
    rpc_method_handlers = {'train': grpc.unary_unary_rpc_method_handler(servicer.train, request_deserializer=nn__service__pb2.TrainRequest.FromString, response_serializer=nn__service__pb2.TrainResponse.SerializeToString), 'evaluate': grpc.unary_unary_rpc_method_handler(servicer.evaluate, request_deserializer=nn__service__pb2.EvaluateRequest.FromString, response_serializer=nn__service__pb2.EvaluateResponse.SerializeToString), 'predict': grpc.unary_unary_rpc_method_handler(servicer.predict, request_deserializer=nn__service__pb2.PredictRequest.FromString, response_serializer=nn__service__pb2.PredictResponse.SerializeToString), 'upload_meta': grpc.unary_unary_rpc_method_handler(servicer.upload_meta, request_deserializer=nn__service__pb2.UploadMetaRequest.FromString, response_serializer=nn__service__pb2.UploadMetaResponse.SerializeToString), 'upload_file': grpc.stream_unary_rpc_method_handler(servicer.upload_file, request_deserializer=nn__service__pb2.ByteChunk.FromString, response_serializer=nn__service__pb2.UploadMetaResponse.SerializeToString), 'save_server_model': grpc.unary_unary_rpc_method_handler(servicer.save_server_model, request_deserializer=nn__service__pb2.SaveModelRequest.FromString, response_serializer=nn__service__pb2.SaveModelResponse.SerializeToString), 'load_server_model': grpc.unary_unary_rpc_method_handler(servicer.load_server_model, request_deserializer=nn__service__pb2.LoadModelRequest.FromString, response_serializer=nn__service__pb2.LoadModelResponse.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('nn.NNService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

class NNService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def train(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        return grpc.experimental.unary_unary(request, target, '/nn.NNService/train', nn__service__pb2.TrainRequest.SerializeToString, nn__service__pb2.TrainResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def evaluate(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            print('Hello World!')
        return grpc.experimental.unary_unary(request, target, '/nn.NNService/evaluate', nn__service__pb2.EvaluateRequest.SerializeToString, nn__service__pb2.EvaluateResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def predict(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            i = 10
            return i + 15
        return grpc.experimental.unary_unary(request, target, '/nn.NNService/predict', nn__service__pb2.PredictRequest.SerializeToString, nn__service__pb2.PredictResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def upload_meta(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            while True:
                i = 10
        return grpc.experimental.unary_unary(request, target, '/nn.NNService/upload_meta', nn__service__pb2.UploadMetaRequest.SerializeToString, nn__service__pb2.UploadMetaResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def upload_file(request_iterator, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            print('Hello World!')
        return grpc.experimental.stream_unary(request_iterator, target, '/nn.NNService/upload_file', nn__service__pb2.ByteChunk.SerializeToString, nn__service__pb2.UploadMetaResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def save_server_model(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        return grpc.experimental.unary_unary(request, target, '/nn.NNService/save_server_model', nn__service__pb2.SaveModelRequest.SerializeToString, nn__service__pb2.SaveModelResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def load_server_model(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            return 10
        return grpc.experimental.unary_unary(request, target, '/nn.NNService/load_server_model', nn__service__pb2.LoadModelRequest.SerializeToString, nn__service__pb2.LoadModelResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)