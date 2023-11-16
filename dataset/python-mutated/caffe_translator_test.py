from google.protobuf import text_format
import numpy as np
import os
import sys
CAFFE_FOUND = False
try:
    from caffe.proto import caffe_pb2
    from caffe2.python import caffe_translator
    CAFFE_FOUND = True
except Exception as e:
    if "'caffe'" in str(e):
        print('PyTorch/Caffe2 now requires a separate installation of caffe. Right now, this is not found, so we will skip the caffe translator test.')
from caffe2.python import utils, workspace, test_util
import unittest

def setUpModule():
    if False:
        i = 10
        return i + 15
    if not (CAFFE_FOUND and os.path.exists('data/testdata/caffe_translator')):
        return
    caffenet = caffe_pb2.NetParameter()
    caffenet_pretrained = caffe_pb2.NetParameter()
    with open('data/testdata/caffe_translator/deploy.prototxt') as f:
        text_format.Merge(f.read(), caffenet)
    with open('data/testdata/caffe_translator/bvlc_reference_caffenet.caffemodel') as f:
        caffenet_pretrained.ParseFromString(f.read())
    for remove_legacy_pad in [True, False]:
        (net, pretrained_params) = caffe_translator.TranslateModel(caffenet, caffenet_pretrained, is_test=True, remove_legacy_pad=remove_legacy_pad)
        with open('data/testdata/caffe_translator/bvlc_reference_caffenet.translatedmodel', 'w') as fid:
            fid.write(str(net))
        for param in pretrained_params.protos:
            workspace.FeedBlob(param.name, utils.Caffe2TensorToNumpyArray(param))
        data = np.load('data/testdata/caffe_translator/data_dump.npy').astype(np.float32)
        workspace.FeedBlob('data', data)
        workspace.RunNetOnce(net.SerializeToString())

@unittest.skipIf(not CAFFE_FOUND, 'No Caffe installation found.')
@unittest.skipIf(not os.path.exists('data/testdata/caffe_translator'), 'No testdata existing for the caffe translator test. Exiting.')
class TestNumericalEquivalence(test_util.TestCase):

    def testBlobs(self):
        if False:
            i = 10
            return i + 15
        names = ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
        for name in names:
            print('Verifying {}'.format(name))
            caffe2_result = workspace.FetchBlob(name)
            reference = np.load('data/testdata/caffe_translator/' + name + '_dump.npy')
            self.assertEqual(caffe2_result.shape, reference.shape)
            scale = np.max(caffe2_result)
            np.testing.assert_almost_equal(caffe2_result / scale, reference / scale, decimal=5)
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('If you do not explicitly ask to run this test, I will not run it. Pass in any argument to have the test run for you.')
        sys.exit(0)
    unittest.main()