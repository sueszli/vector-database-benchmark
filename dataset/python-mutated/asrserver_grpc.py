"""
@author: nl8590687
ASRT语音识别基于gRPC协议的API服务器程序
"""
import argparse
import time
from concurrent import futures
import grpc
from assets.asrt_pb2_grpc import AsrtGrpcServiceServicer, add_AsrtGrpcServiceServicer_to_server
from assets.asrt_pb2 import SpeechResponse, TextResponse
from speech_model import ModelSpeech
from model_zoo.speech_model.keras_backend import SpeechModel251BN
from speech_features import Spectrogram
from language_model3 import ModelLanguage
from utils.ops import decode_wav_bytes
API_STATUS_CODE_OK = 200000
API_STATUS_CODE_OK_PART = 206000
API_STATUS_CODE_CLIENT_ERROR = 400000
API_STATUS_CODE_CLIENT_ERROR_FORMAT = 400001
API_STATUS_CODE_CLIENT_ERROR_CONFIG = 400002
API_STATUS_CODE_SERVER_ERROR = 500000
API_STATUS_CODE_SERVER_ERROR_RUNNING = 500001
parser = argparse.ArgumentParser(description='ASRT gRPC Protocol API Service')
parser.add_argument('--listen', default='0.0.0.0', type=str, help='the network to listen')
parser.add_argument('--port', default='20002', type=str, help='the port to listen')
args = parser.parse_args()
AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
OUTPUT_SIZE = 1428
sm251bn = SpeechModel251BN(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS), output_size=OUTPUT_SIZE)
feat = Spectrogram()
ms = ModelSpeech(sm251bn, feat, max_label_length=64)
ms.load_model('save_models/' + sm251bn.get_model_name() + '.model.h5')
ml = ModelLanguage('model_language')
ml.load_model()
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class ApiService(AsrtGrpcServiceServicer):
    """
    继承AsrtGrpcServiceServicer,实现hello方法
    """

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def Speech(self, request, context):
        if False:
            print('Hello World!')
        '\n        具体实现Speech的方法, 并按照pb的返回对象构造SpeechResponse返回\n        :param request:\n        :param context:\n        :return:\n        '
        wav_data = request.wav_data
        wav_samples = decode_wav_bytes(samples_data=wav_data.samples, channels=wav_data.channels, byte_width=wav_data.byte_width)
        result = ms.recognize_speech(wav_samples, wav_data.sample_rate)
        print('语音识别声学模型结果:', result)
        return SpeechResponse(status_code=API_STATUS_CODE_OK, status_message='', result_data=result)

    def Language(self, request, context):
        if False:
            print('Hello World!')
        '\n        具体实现Language的方法, 并按照pb的返回对象构造TextResponse返回\n        :param request:\n        :param context:\n        :return:\n        '
        print('Language收到了请求:', request)
        result = ml.pinyin_to_text(list(request.pinyins))
        print('Language结果:', result)
        return TextResponse(status_code=API_STATUS_CODE_OK, status_message='', text_result=result)

    def All(self, request, context):
        if False:
            while True:
                i = 10
        '\n        具体实现All的方法, 并按照pb的返回对象构造TextResponse返回\n        :param request:\n        :param context:\n        :return:\n        '
        wav_data = request.wav_data
        wav_samples = decode_wav_bytes(samples_data=wav_data.samples, channels=wav_data.channels, byte_width=wav_data.byte_width)
        result_speech = ms.recognize_speech(wav_samples, wav_data.sample_rate)
        result = ml.pinyin_to_text(result_speech)
        print('语音识别结果:', result)
        return TextResponse(status_code=API_STATUS_CODE_OK, status_message='', text_result=result)

    def Stream(self, request_iterator, context):
        if False:
            i = 10
            return i + 15
        '\n        具体实现Stream的方法, 并按照pb的返回对象构造TextResponse返回\n        :param request_iterator:\n        :param context:\n        :return:\n        '
        result = list()
        tmp_result_last = list()
        beam_size = 100
        for request in request_iterator:
            wav_data = request.wav_data
            wav_samples = decode_wav_bytes(samples_data=wav_data.samples, channels=wav_data.channels, byte_width=wav_data.byte_width)
            result_speech = ms.recognize_speech(wav_samples, wav_data.sample_rate)
            for item_pinyin in result_speech:
                tmp_result = ml.pinyin_stream_decode(tmp_result_last, item_pinyin, beam_size)
                if len(tmp_result) == 0 and len(tmp_result_last) > 0:
                    result.append(tmp_result_last[0][0])
                    print('流式语音识别结果：', ''.join(result))
                    yield TextResponse(status_code=API_STATUS_CODE_OK, status_message='', text_result=''.join(result))
                    result = list()
                    tmp_result = ml.pinyin_stream_decode([], item_pinyin, beam_size)
                tmp_result_last = tmp_result
                yield TextResponse(status_code=API_STATUS_CODE_OK_PART, status_message='', text_result=''.join(tmp_result[0][0]))
        if len(tmp_result_last) > 0:
            result.append(tmp_result_last[0][0])
            print('流式语音识别结果：', ''.join(result))
            yield TextResponse(status_code=API_STATUS_CODE_OK, status_message='', text_result=''.join(result))

def run(host, port):
    if False:
        while True:
            i = 10
    '\n    gRPC API服务启动\n    :return:\n    '
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_AsrtGrpcServiceServicer_to_server(ApiService(), server)
    server.add_insecure_port(''.join([host, ':', port]))
    server.start()
    print('start service...')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
if __name__ == '__main__':
    run(host=args.listen, port=args.port)