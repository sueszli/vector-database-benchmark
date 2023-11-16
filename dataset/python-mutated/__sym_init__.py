import os
from caffe2.proto import caffe2_pb2

def _parseFile(filename):
    if False:
        return 10
    out_net = caffe2_pb2.NetDef()
    dir_path = os.path.dirname(__file__)
    with open('{dir_path}/{filename}'.format(dir_path=dir_path, filename=filename), 'rb') as f:
        out_net.ParseFromString(f.read())
    return out_net
init_net = _parseFile('init_net.pb')
predict_net = _parseFile('predict_net.pb')