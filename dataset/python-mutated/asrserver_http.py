"""
@author: nl8590687
ASRT语音识别基于HTTP协议的API服务器程序
"""
import argparse
import base64
import json
from flask import Flask, Response, request
from speech_model import ModelSpeech
from model_zoo.speech_model.keras_backend import SpeechModel251BN
from speech_features import Spectrogram
from language_model3 import ModelLanguage
from utils.ops import decode_wav_bytes
API_STATUS_CODE_OK = 200000
API_STATUS_CODE_CLIENT_ERROR = 400000
API_STATUS_CODE_CLIENT_ERROR_FORMAT = 400001
API_STATUS_CODE_CLIENT_ERROR_CONFIG = 400002
API_STATUS_CODE_SERVER_ERROR = 500000
API_STATUS_CODE_SERVER_ERROR_RUNNING = 500001
parser = argparse.ArgumentParser(description='ASRT HTTP+Json RESTful API Service')
parser.add_argument('--listen', default='0.0.0.0', type=str, help='the network to listen')
parser.add_argument('--port', default='20001', type=str, help='the port to listen')
args = parser.parse_args()
app = Flask('ASRT API Service')
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

class AsrtApiResponse:
    """
    ASRT语音识别基于HTTP协议的API接口响应类
    """

    def __init__(self, status_code, status_message='', result=''):
        if False:
            while True:
                i = 10
        self.status_code = status_code
        self.status_message = status_message
        self.result = result

    def to_json(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        类转json\n        '
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

@app.route('/', methods=['GET'])
def index_get():
    if False:
        i = 10
        return i + 15
    '\n    根路径handle GET方法\n    '
    buffer = ''
    with open('assets/default.html', 'r', encoding='utf-8') as file_handle:
        buffer = file_handle.read()
    return Response(buffer, mimetype='text/html; charset=utf-8')

@app.route('/', methods=['POST'])
def index_post():
    if False:
        print('Hello World!')
    '\n    根路径handle POST方法\n    '
    json_data = AsrtApiResponse(API_STATUS_CODE_OK, 'ok')
    buffer = json_data.to_json()
    return Response(buffer, mimetype='application/json')

@app.route('/<level>', methods=['POST'])
def recognition_post(level):
    if False:
        while True:
            i = 10
    '\n    其他路径 POST方法\n    '
    try:
        if level == 'speech':
            request_data = request.get_json()
            samples = request_data['samples']
            wavdata_bytes = base64.urlsafe_b64decode(bytes(samples, encoding='utf-8'))
            sample_rate = request_data['sample_rate']
            channels = request_data['channels']
            byte_width = request_data['byte_width']
            wavdata = decode_wav_bytes(samples_data=wavdata_bytes, channels=channels, byte_width=byte_width)
            result = ms.recognize_speech(wavdata, sample_rate)
            json_data = AsrtApiResponse(API_STATUS_CODE_OK, 'speech level')
            json_data.result = result
            buffer = json_data.to_json()
            print('output:', buffer)
            return Response(buffer, mimetype='application/json')
        elif level == 'language':
            request_data = request.get_json()
            seq_pinyin = request_data['sequence_pinyin']
            result = ml.pinyin_to_text(seq_pinyin)
            json_data = AsrtApiResponse(API_STATUS_CODE_OK, 'language level')
            json_data.result = result
            buffer = json_data.to_json()
            print('output:', buffer)
            return Response(buffer, mimetype='application/json')
        elif level == 'all':
            request_data = request.get_json()
            samples = request_data['samples']
            wavdata_bytes = base64.urlsafe_b64decode(samples)
            sample_rate = request_data['sample_rate']
            channels = request_data['channels']
            byte_width = request_data['byte_width']
            wavdata = decode_wav_bytes(samples_data=wavdata_bytes, channels=channels, byte_width=byte_width)
            result_speech = ms.recognize_speech(wavdata, sample_rate)
            result = ml.pinyin_to_text(result_speech)
            json_data = AsrtApiResponse(API_STATUS_CODE_OK, 'all level')
            json_data.result = result
            buffer = json_data.to_json()
            print('ASRT Result:', result, 'output:', buffer)
            return Response(buffer, mimetype='application/json')
        else:
            request_data = request.get_json()
            print('input:', request_data)
            json_data = AsrtApiResponse(API_STATUS_CODE_CLIENT_ERROR, '')
            buffer = json_data.to_json()
            print('output:', buffer)
            return Response(buffer, mimetype='application/json')
    except Exception as except_general:
        request_data = request.get_json()
        json_data = AsrtApiResponse(API_STATUS_CODE_SERVER_ERROR, str(except_general))
        buffer = json_data.to_json()
        print('output:', buffer, 'error:', except_general)
        return Response(buffer, mimetype='application/json')
if __name__ == '__main__':
    import waitress
    waitress.serve(app, host=args.listen, port=args.port)