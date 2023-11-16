import os
import tempfile
import unittest
import paddle
from paddle.inference import PlaceType, PrecisionType, convert_to_mixed_precision
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.vision.models import resnet50

@unittest.skipIf(not paddle.is_compiled_with_cuda() or paddle.get_cudnn_version() < 8000, 'should compile with cuda.')
class TestConvertToMixedPrecision(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()
        model = resnet50(True)
        net = to_static(model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')])
        paddle.jit.save(net, os.path.join(self.temp_dir.name, 'resnet50/inference'))

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def test_convert_to_mixed_precision(self):
        if False:
            return 10
        mixed_precision_options = [PrecisionType.Half, PrecisionType.Half, PrecisionType.Half, PrecisionType.Bfloat16]
        keep_io_types_options = [True, False, False, True]
        black_list_options = [set(), set(), {'conv2d'}, set()]
        test_configs = zip(mixed_precision_options, keep_io_types_options, black_list_options)
        for (mixed_precision, keep_io_types, black_list) in test_configs:
            config = f'mixed_precision={mixed_precision}-keep_io_types={keep_io_types}-black_list={black_list}'
            with self.subTest(mixed_precision=mixed_precision, keep_io_types=keep_io_types, black_list=black_list):
                convert_to_mixed_precision(os.path.join(self.temp_dir.name, 'resnet50/inference.pdmodel'), os.path.join(self.temp_dir.name, 'resnet50/inference.pdiparams'), os.path.join(self.temp_dir.name, f'{config}/inference.pdmodel'), os.path.join(self.temp_dir.name, f'{config}/inference.pdiparams'), backend=PlaceType.GPU, mixed_precision=mixed_precision, keep_io_types=keep_io_types, black_list=black_list)
if __name__ == '__main__':
    unittest.main()