from bigdl.ppml.utils.safepickle import SafePickle
from bigdl.ppml.fl.nn.fl_client import FLClient
from bigdl.ppml.fl.nn.generated.nn_service_pb2 import TrainRequest, PredictRequest, UploadMetaRequest
from bigdl.ppml.fl.nn.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.nn.utils import ndarray_map_to_tensor_map
import threading
from bigdl.dllib.utils.log4Error import invalidInputError, invalidOperationError
from bigdl.ppml.fl.nn.utils import ClassAndArgsWrapper

class NNClient(object):
    _lock = threading.Lock()

    def __init__(self, aggregator) -> None:
        if False:
            print('Hello World!')
        if FLClient.channel is None:
            invalidOperationError(False, 'No channel found, please make sure you called                 init_fl_context()')
        if FLClient.client_id is None:
            invalidOperationError(False, 'You have to set client_id with integer like:                 init_fl_context(client_id=1)')
        self.nn_stub = NNServiceStub(FLClient.channel)
        self.client_uuid = FLClient.client_id
        self.aggregator = aggregator

    def train(self, x):
        if False:
            for i in range(10):
                print('nop')
        tensor_map = ndarray_map_to_tensor_map(x)
        train_request = TrainRequest(clientuuid=self.client_uuid, data=tensor_map, algorithm=self.aggregator)
        response = self.nn_stub.train(train_request)
        if response.code == 1:
            invalidInputError(False, response.response)
        return response

    def predict(self, x):
        if False:
            for i in range(10):
                print('nop')
        tensor_map = ndarray_map_to_tensor_map(x)
        predict_request = PredictRequest(clientuuid=self.client_uuid, data=tensor_map, algorithm=self.aggregator)
        response = self.nn_stub.predict(predict_request)
        if response.code == 1:
            invalidInputError(False, response.response)
        return response

    def upload_meta(self, loss_fn, optimizer_cls, optimizer_args):
        if False:
            return 10
        loss_fn = SafePickle.dumps(loss_fn)
        optimizer = ClassAndArgsWrapper(optimizer_cls, optimizer_args).to_protobuf()
        request = UploadMetaRequest(client_uuid=self.client_uuid, loss_fn=loss_fn, optimizer=optimizer, aggregator=self.aggregator)
        return self.nn_stub.upload_meta(request)