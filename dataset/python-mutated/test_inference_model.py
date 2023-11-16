import os
import pytest
import numpy as np
from bigdl.dllib.feature.dataset.base import maybe_download
from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.orca.inference import InferenceModel
import tarfile
np.random.seed(1337)
resource_path = os.path.join(os.path.dirname(__file__), '../resources')
property_path = os.path.join(os.path.dirname(__file__), '../../../../../../scala/target/classes/app.properties')
data_url = 'https://s3-ap-southeast-1.amazonaws.com'
with open(property_path) as f:
    for _ in range(2):
        next(f)
    for line in f:
        if 'data-store-url' in line:
            line = line.strip()
            data_url = line.split('=')[1].replace('\\', '')

class TestInferenceModel(ZooTestCase):

    def test_load_bigdl(self):
        if False:
            i = 10
            return i + 15
        model = InferenceModel(3)
        model.load_bigdl(os.path.join(resource_path, 'models/bigdl/bigdl_lenet.model'))
        input_data = np.random.random([4, 28, 28, 1])
        output_data = model.predict(input_data)

    def test_load_caffe(self):
        if False:
            return 10
        model = InferenceModel(10)
        model.load_caffe(os.path.join(resource_path, 'models/caffe/test_persist.prototxt'), os.path.join(resource_path, 'models/caffe/test_persist.caffemodel'))
        input_data = np.random.random([4, 3, 8, 8])
        output_data = model.predict(input_data)

    def test_load_openvino(self):
        if False:
            i = 10
            return i + 15
        local_path = self.create_temp_dir()
        model = InferenceModel(1)
        model_url = data_url + '/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.xml'
        weight_url = data_url + '/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.bin'
        model_path = maybe_download('resnet_v1_50.xml', local_path, model_url)
        weight_path = maybe_download('resnet_v1_50.bin', local_path, weight_url)
        model.load_openvino(model_path, weight_path)
        input_data = np.random.random([4, 1, 224, 224, 3])
        model.predict(input_data)
if __name__ == '__main__':
    pytest.main([__file__])