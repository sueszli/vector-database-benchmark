import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'cpp'
from caffe2.proto import caffe2_pb2, metanet_pb2
import unittest

class TestCrossProtoCalls(unittest.TestCase):

    def testSimple(self):
        if False:
            while True:
                i = 10
        net = caffe2_pb2.NetDef()
        meta = metanet_pb2.MetaNetDef()
        meta.nets.add(key='foo', value=net)