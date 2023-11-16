"""Concurrency tests for quantize_model."""
from concurrent import futures
import numpy as np
import tensorflow
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import quantize_model
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import autotrackable

class MultiThreadedTest(test.TestCase):
    """Tests involving multiple threads."""

    def setUp(self):
        if False:
            while True:
                i = 10
        super(MultiThreadedTest, self).setUp()
        self.pool = futures.ThreadPoolExecutor(max_workers=4)

    def _convert_with_calibration(self):
        if False:
            return 10

        class ModelWithAdd(autotrackable.AutoTrackable):
            """Basic model with addition."""

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=[10], dtype=dtypes.float32, name='x'), tensor_spec.TensorSpec(shape=[10], dtype=dtypes.float32, name='y')])
            def add(self, x, y):
                if False:
                    while True:
                        i = 10
                res = math_ops.add(x, y)
                return {'output': res}

        def data_gen():
            if False:
                for i in range(10):
                    print('nop')
            for _ in range(255):
                yield {'x': ops.convert_to_tensor(np.random.uniform(size=10).astype('f4')), 'y': ops.convert_to_tensor(np.random.uniform(size=10).astype('f4'))}
        root = ModelWithAdd()
        temp_path = self.create_tempdir().full_path
        saved_model_save.save(root, temp_path, signatures=root.add.get_concrete_function())
        quantization_options = quant_opts_pb2.QuantizationOptions(quantization_method=quant_opts_pb2.QuantizationMethod(preset_method=quant_opts_pb2.QuantizationMethod.PresetMethod.METHOD_STATIC_RANGE_INT8), tags={tag_constants.SERVING}, signature_keys=['serving_default'])
        model = quantize_model.quantize(temp_path, quantization_options=quantization_options, representative_dataset=data_gen())
        return model

    @test_util.run_in_graph_and_eager_modes
    def test_multiple_conversion_jobs_with_calibration(self):
        if False:
            for i in range(10):
                print('nop')
        with self.pool:
            jobs = []
            for _ in range(10):
                jobs.append(self.pool.submit(self._convert_with_calibration))
            for job in jobs:
                self.assertIsNotNone(job.result())
if __name__ == '__main__':
    test.main()