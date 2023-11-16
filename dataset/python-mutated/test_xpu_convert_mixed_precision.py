import os
import tempfile
import unittest
import paddle
from paddle.inference import PlaceType, PrecisionType, convert_to_mixed_precision
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.vision.models import resnet50

class ConvertMixedPrecison(unittest.TestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir = tempfile.TemporaryDirectory()
        model = resnet50(True)
        net = to_static(model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')])
        paddle.jit.save(net, os.path.join(self.temp_dir.name, 'resnet50/inference'))
        convert_to_mixed_precision(os.path.join(self.temp_dir.name, 'resnet50/inference.pdmodel'), os.path.join(self.temp_dir.name, 'resnet50/inference.pdiparams'), os.path.join(self.temp_dir.name, 'mixed_precision/inference.pdmodel'), os.path.join(self.temp_dir.name, 'mixed_precision/inference.pdiparams'), backend=PlaceType.XPU, mixed_precision=PrecisionType.Half)
        self.temp_dir.cleanup()
if __name__ == '__main__':
    unittest.main()