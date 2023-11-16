from omnihub.frameworks.huggingface import HuggingFaceModelHub
from omnihub.frameworks.keras import KerasModelHub
from omnihub.frameworks.onnx import OnnxModelHub
from omnihub.frameworks.pytorch import PytorchModelHub
from omnihub.frameworks.tensorflow import TensorflowModelHub
import os
from omnihub.model_hub import omnihub_dir


def test_keras():
    keras_model_hub = KerasModelHub()
    model_path = keras_model_hub.download_model('vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    keras_model_hub.stage_model(model_path, 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    assert os.path.exists(os.path.join(omnihub_dir, 'keras', 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'))


def test_onnx():
    onnx_model_hub = OnnxModelHub()
    onnx_model_hub.download_model('vision/body_analysis/age_gender/models/age_googlenet.onnx')
    assert os.path.exists(os.path.join(omnihub_dir, 'onnx', 'age_googlenet.onnx'))


def test_tensorflow():
    # https://tfhub.dev/emilutz/vgg19-block4-conv2-unpooling-decoder/1?tf-hub-format=compressed
    tensorflow_model_hub = TensorflowModelHub()
    tensorflow_model_hub.download_model('emilutz/vgg19-block4-conv2-unpooling-decoder/1')


def test_pytorch():
    pytorch_model_hub = PytorchModelHub()
    pytorch_model_hub.download_model('resnet18')
    assert os.path.exists(os.path.join(omnihub_dir, 'pytorch', 'resnet18.onnx'))


def test_huggingface():
    huggingface_model_hub = HuggingFaceModelHub()
    huggingface_model_hub.download_model('gpt2',framework_name='pytorch')
    #assert os.path.exists(os.path.join(omnihub_dir, 'huggingface', 'tf_model.h5'))
