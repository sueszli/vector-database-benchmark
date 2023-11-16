from threading import Condition
import threading
import torch
import bigdl.ppml.fl.nn.pytorch.aggregator as pt_agg
import bigdl.ppml.fl.nn.tensorflow.aggregator as tf_agg
from bigdl.ppml.fl.nn.generated.fl_base_pb2 import TensorMap
from bigdl.ppml.fl.nn.generated.nn_service_pb2 import LoadModelResponse, SaveModelResponse, TrainRequest, TrainResponse, PredictResponse, UploadMetaResponse
from bigdl.ppml.fl.nn.generated.nn_service_pb2_grpc import *
from bigdl.ppml.fl.nn.utils import tensor_map_to_ndarray_map
import tensorflow as tf
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.ppml.utils.safepickle import SafePickle
import tempfile
import traceback
import os
import logging

class NNServiceImpl(NNServiceServicer):

    def __init__(self, conf, **kargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.client_num = conf['clientNum']
        self.aggregator_map = {'tf': tf_agg.Aggregator(conf, **kargs), 'pt': pt_agg.Aggregator(conf, **kargs)}
        self.model_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.model_dir, 'vfl_server_model')
        self.client_num = int(self.client_num)
        self.condition = Condition()
        self._lock = threading.Lock()

    def train(self, request: TrainRequest, context):
        if False:
            for i in range(10):
                print('nop')
        tensor_map = request.data.tensorMap
        client_id = request.clientuuid
        self.validate_client_id(client_id)
        ndarray_map = tensor_map_to_ndarray_map(tensor_map)
        aggregator = self.aggregator_map[request.algorithm]
        try:
            aggregator.put_client_data(client_id, ndarray_map, 'train')
            msg = f'[client {client_id} batch trained]'
            code = 0
        except Exception as e:
            msg = traceback.format_exc()
            logging.error(msg)
            code = 1
        return TrainResponse(response=msg, data=aggregator.server_data['train'][client_id], code=code)

    def evaluate(self, request, context):
        if False:
            i = 10
            return i + 15
        return super().evaluate(request, context)

    def predict(self, request, context):
        if False:
            return 10
        tensor_map = request.data.tensorMap
        client_id = request.clientuuid
        self.validate_client_id(client_id)
        ndarray_map = tensor_map_to_ndarray_map(tensor_map)
        aggregator = self.aggregator_map[request.algorithm]
        try:
            aggregator.put_client_data(client_id, ndarray_map, 'pred')
            msg = f'[client {client_id} batch predicted]'
            code = 0
        except Exception as e:
            msg = traceback.format_exc()
            logging.error(msg)
            code = 1
        return PredictResponse(response=msg, data=aggregator.server_data['pred'][client_id], code=code)

    def upload_meta(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        try:
            loss_fn = SafePickle.loads(request.loss_fn)
            client_id = request.client_uuid
            aggregator = self.aggregator_map[request.aggregator]
            aggregator.load_uploaded_model(client_id, self.model_path)
            aggregator.set_meta(loss_fn, request.optimizer)
            msg = 'Upload meta success, server model is ready.'
            code = 0
        except Exception as e:
            msg = traceback.format_exc()
            logging.error(msg)
            code = 1
        return UploadMetaResponse(message=msg, code=code)

    def upload_file(self, request_iterator, context):
        if False:
            for i in range(10):
                print('nop')
        try:
            logging.debug('Trying to acquire lock of server model')
            self.condition.acquire()
            logging.debug('Acquired lock of server model')
            if os.path.exists(self.model_path):
                logging.warn('Model file exists, will not upload.')
                msg = 'Model file exists, will not upload, exiting..'
                code = 1
            else:
                logging.debug('Unpacking file chunk from FLClient')
                with open(self.model_path, 'wb') as f:
                    for byte_chunk in request_iterator:
                        f.write(byte_chunk.buffer)
                logging.info(f'Server received model file, length: {os.path.getsize(self.model_path)}')
            self.condition.release()
            msg = 'Upload model file sucess'
            code = 0
        except Exception as e:
            traceback.print_exc()
            msg = traceback.format_exc()
            code = 1
        return UploadMetaResponse(message=msg, code=code)

    def validate_client_id(self, client_id):
        if False:
            while True:
                i = 10
        try:
            client_id = int(client_id)
        except:
            invalidInputError(False, f'client ID must be a number, got {client_id}')
        if client_id <= 0 or client_id > self.client_num:
            invalidInputError(False, f'invalid client ID received: {client_id},                 must be in range of client number [1, {self.client_num}]')

    def save_server_model(self, request, context):
        if False:
            i = 10
            return i + 15
        aggregator = self.aggregator_map[request.backend]
        aggregator.save_server_model(request.model_path)
        return SaveModelResponse(message=f'Server model saved to {request.model_path}')

    def load_server_model(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        aggregator = self.aggregator_map[request.backend]
        aggregator.load_server_model(request.client_id, request.model_path)
        return LoadModelResponse(message=f'Server model loaded from {request.model_path}')