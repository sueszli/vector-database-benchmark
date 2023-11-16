"""
@author: nl8590687
ASRT语音识别asrserver grpc协议测试专用客户端
"""
import grpc
import time
from assets.asrt_pb2_grpc import AsrtGrpcServiceStub
from assets.asrt_pb2 import SpeechRequest, LanguageRequest, WavData
from utils.ops import read_wav_bytes

def run_speech():
    if False:
        for i in range(10):
            print('nop')
    '\n    请求ASRT服务Speech方法\n    :return:\n    '
    conn = grpc.insecure_channel('127.0.0.1:20002')
    client = AsrtGrpcServiceStub(channel=conn)
    (wav_bytes, sample_rate, channels, sample_width) = read_wav_bytes('assets/A11_0.wav')
    print('sample_width:', sample_width)
    wav_data = WavData(samples=wav_bytes, sample_rate=sample_rate, channels=channels, byte_width=sample_width)
    request = SpeechRequest(wav_data=wav_data)
    time_stamp0 = time.time()
    response = client.Speech(request)
    time_stamp1 = time.time()
    print('time:', time_stamp1 - time_stamp0, 's')
    print('received:', response.result_data)

def run_lan():
    if False:
        return 10
    '\n    请求ASRT服务Language方法\n    :return:\n    '
    conn = grpc.insecure_channel('127.0.0.1:20002')
    client = AsrtGrpcServiceStub(channel=conn)
    pinyin_data = ['ni3', 'hao3', 'ya5']
    request = LanguageRequest(pinyins=pinyin_data)
    time_stamp0 = time.time()
    response = client.Language(request)
    time_stamp1 = time.time()
    print('time:', time_stamp1 - time_stamp0, 's')
    print('received:', response.text_result)

def run_all():
    if False:
        return 10
    '\n    请求ASRT服务All方法\n    :return:\n    '
    conn = grpc.insecure_channel('127.0.0.1:20002')
    client = AsrtGrpcServiceStub(channel=conn)
    (wav_bytes, sample_rate, channels, sample_width) = read_wav_bytes('assets/A11_0.wav')
    print('sample_width:', sample_width)
    wav_data = WavData(samples=wav_bytes, sample_rate=sample_rate, channels=channels, byte_width=sample_width)
    request = SpeechRequest(wav_data=wav_data)
    time_stamp0 = time.time()
    response = client.All(request)
    time_stamp1 = time.time()
    print('received:', response.text_result)
    print('time:', time_stamp1 - time_stamp0, 's')

def run_stream():
    if False:
        return 10
    '\n    请求ASRT服务Stream方法\n    :return:\n    '
    conn = grpc.insecure_channel('127.0.0.1:20002')
    client = AsrtGrpcServiceStub(channel=conn)
    (wav_bytes, sample_rate, channels, sample_width) = read_wav_bytes('assets/A11_0.wav')
    print('sample_width:', sample_width)
    wav_data = WavData(samples=wav_bytes, sample_rate=sample_rate, channels=channels, byte_width=sample_width)

    def make_some_data():
        if False:
            for i in range(10):
                print('nop')
        for _ in range(1):
            time.sleep(1)
            yield SpeechRequest(wav_data=wav_data)
    try:
        status_response = client.Stream(make_some_data())
        for ret in status_response:
            print('received:', ret.text_result, ' , status:', ret.status_code)
            time.sleep(0.1)
    except Exception as any_exception:
        print(f'err in send_status:{any_exception}')
        return
if __name__ == '__main__':
    run_stream()