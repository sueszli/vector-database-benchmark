import logging
import grpc
from bigdl.ppml.fl.nn.generated.nn_service_pb2 import TrainRequest, PredictRequest, UploadMetaRequest
from bigdl.ppml.fl.nn.generated.nn_service_pb2_grpc import *
import yaml
import threading
from bigdl.dllib.utils.log4Error import invalidInputError

class FLClient(object):
    channel = None
    _lock = threading.Lock()
    client_id = None
    target = 'localhost:8980'
    secure = False
    creds = None

    @staticmethod
    def set_client_id(client_id):
        if False:
            i = 10
            return i + 15
        FLClient.client_id = client_id

    @staticmethod
    def set_target(target):
        if False:
            return 10
        FLClient.target = target

    @staticmethod
    def ensure_initialized():
        if False:
            for i in range(10):
                print('nop')
        with FLClient._lock:
            if FLClient.channel == None:
                if FLClient.secure:
                    FLClient.channel = grpc.secure_channel(FLClient.target, FLClient.creds)
                else:
                    FLClient.channel = grpc.insecure_channel(FLClient.target)

    @staticmethod
    def load_config():
        if False:
            print('Hello World!')
        try:
            with open('ppml-conf.yaml', 'r') as stream:
                conf = yaml.safe_load(stream)
                if 'privateKeyFilePath' in conf:
                    FLClient.secure = True
                    with open(conf['privateKeyFilePath'], 'rb') as f:
                        FLClient.creds = grpc.ssl_channel_credentials(f.read())
        except yaml.YAMLError as e:
            logging.warn('Loading config failed, using default config ')
        except Exception as e:
            logging.warn('Failed to find config file "ppml-conf.yaml", using default config')