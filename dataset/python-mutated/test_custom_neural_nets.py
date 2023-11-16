import os
import shutil
import tempfile
import unittest
import numpy as np
import coremltools
import coremltools.models.datatypes as datatypes
from coremltools.models import neural_network as neural_network
from coremltools.models.utils import _macos_version, _is_macos

class SimpleTest(unittest.TestCase):

    def test_fixed_seq_len(self):
        if False:
            return 10
        '\n        Input has a fixed sequence length.\n        (this happens when model is trained using padded sequences, inspiration: https://forums.developer.apple.com/thread/80407)\n\n        (Seq,Batch,C,H,W)\n        embedding: input shape (15,1,1,1,1) --> output shape (15,1,32,1,1)\n        permute  : input shape (15,1,32,1,1) --> output shape (1,1,32,1,15)\n        flatten  : input shape (1,1,32,1,15) --> output shape (1,1,32 * 15,1,1)\n        dense    : input shape (1,1,480,1,1) --> output shape (1,1,2,1,1)\n        '
        coreml_preds = []
        input_dim = (1, 1, 1)
        output_dim = (1, 1, 1)
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', datatypes.Array(*output_dim))]
        builder = neural_network.NeuralNetworkBuilder(input_features, output_features)
        builder.add_embedding('embed', W=np.random.rand(10, 32), b=None, input_dim=10, output_channels=32, has_bias=0, input_name='data', output_name='embed')
        builder.add_permute('permute', dim=[3, 1, 2, 0], input_name='embed', output_name='permute')
        builder.add_flatten('flatten', mode=0, input_name='permute', output_name='flatten')
        builder.add_inner_product('dense', W=np.random.rand(480, 2), b=None, input_channels=480, output_channels=2, has_bias=0, input_name='flatten', output_name='output')
        del builder.spec.description.output[-1]
        output = builder.spec.description.output.add()
        output.name = 'output'
        output.type.multiArrayType.dataType = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Value('DOUBLE')
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, 'test_layer.mlmodel')
        coremltools.utils.save_spec(builder.spec, model_path)
        coreml_model = coremltools.models.MLModel(model_path)
        X = np.random.randint(low=0, high=10, size=15)
        X = np.reshape(X, (15, 1, 1, 1, 1)).astype(np.float32)
        coreml_input = {'data': X}
        if _is_macos() and _macos_version() >= (10, 13):
            coreml_preds = coreml_model.predict(coreml_input)['output']
            self.assertEquals(len(coreml_preds.flatten()), 2)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)