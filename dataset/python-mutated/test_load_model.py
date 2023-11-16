from __future__ import print_function
import numpy as np
import pytest
from numpy.testing import assert_allclose
import bigdl.dllib.nn.layer as BLayer
from bigdl.dllib.keras.converter import WeightLoader
from bigdl.dllib.keras.converter import DefinitionLoader
np.random.seed(1337)
from test.bigdl.test_utils import BigDLTestCase, TestModels
from bigdl.dllib.nn.keras.keras_utils import *
import keras.backend as K

class TestLoadModel(BigDLTestCase):

    def __kmodel_load_def_weight_test(self, kmodel, input_data):
        if False:
            return 10
        (keras_model_path_json, keras_model_path_hdf5) = dump_keras(kmodel, dump_weights=True)
        bmodel = DefinitionLoader.from_json_path(keras_model_path_json)
        WeightLoader.load_weights_from_hdf5(bmodel, kmodel, keras_model_path_hdf5)
        bmodel.training(False)
        boutput = bmodel.forward(input_data)
        koutput = kmodel.predict(input_data)
        assert_allclose(boutput, koutput, rtol=1e-05)

    def test_load_api_with_hdf5(self):
        if False:
            i = 10
            return i + 15
        K.set_image_dim_ordering('th')
        (kmodel, input_data, output_data) = TestModels.kmodel_graph_1_layer()
        (keras_model_json_path, keras_model_hdf5_path) = dump_keras(kmodel, dump_weights=True)
        bmodel = BLayer.Model.load_keras(json_path=keras_model_json_path, hdf5_path=keras_model_hdf5_path)
        self.assert_allclose(kmodel.predict(input_data), bmodel.forward(input_data))

    def test_load_model_with_hdf5_with_definition(self):
        if False:
            while True:
                i = 10
        (kmodel, input_data, output_data) = TestModels.kmodel_graph_1_layer()
        (keras_model_json_path, keras_model_hdf5_path) = dump_keras(kmodel, dump_weights=True)
        bmodel = BLayer.Model.load_keras(hdf5_path=keras_model_hdf5_path)
        self.assert_allclose(kmodel.predict(input_data), bmodel.forward(input_data))

    def test_load_api_no_hdf5(self):
        if False:
            i = 10
            return i + 15
        K.set_image_dim_ordering('th')
        (kmodel, input_data, output_data) = TestModels.kmodel_graph_1_layer()
        (keras_model_json_path, keras_model_hdf5_path) = dump_keras(kmodel, dump_weights=True)
        bmodel = BLayer.Model.load_keras(json_path=keras_model_json_path)

    def test_load_def_weights_graph_1_layer(self):
        if False:
            i = 10
            return i + 15
        K.set_image_dim_ordering('th')
        (kmodel, input_data, output_data) = TestModels.kmodel_graph_1_layer()
        self.__kmodel_load_def_weight_test(kmodel, input_data)

    def test_load_def_weights_graph_activation(self):
        if False:
            print('Hello World!')
        K.set_image_dim_ordering('th')
        (kmodel, input_data, output_data) = TestModels.kmodel_graph_activation_is_layer()
        self.__kmodel_load_def_weight_test(kmodel, input_data)

    def test_load_def_weights_kmodel_seq_lenet_mnist(self):
        if False:
            i = 10
            return i + 15
        K.set_image_dim_ordering('th')
        (kmodel, input_data, output_data) = TestModels.kmodel_seq_lenet_mnist()
        self.__kmodel_load_def_weight_test(kmodel, input_data)

    def test_load_definition(self):
        if False:
            for i in range(10):
                print('nop')
        K.set_image_dim_ordering('th')
        (kmodel, input_data, output_data) = TestModels.kmodel_seq_lenet_mnist()
        (keras_model_json_path, keras_model_hdf5_path) = dump_keras(kmodel, dump_weights=True)
        bmodel = DefinitionLoader.from_json_path(keras_model_json_path)
        WeightLoader.load_weights_from_kmodel(bmodel, kmodel)
        self.assert_allclose(bmodel.forward(input_data), kmodel.predict(input_data))

    def test_load_weights(self):
        if False:
            i = 10
            return i + 15
        K.set_image_dim_ordering('th')
        (kmodel, input_data, output_data) = TestModels.kmodel_graph_1_layer()
        (keras_model_json_path, keras_model_hdf5_path) = dump_keras(kmodel, dump_weights=True)
        bmodel = DefinitionLoader.from_json_path(keras_model_json_path)
        kmodel.set_weights([kmodel.get_weights()[0] + 100, kmodel.get_weights()[1]])
        WeightLoader.load_weights_from_hdf5(bmodel, kmodel, filepath=keras_model_hdf5_path)
        self.assert_allclose(bmodel.forward(input_data), kmodel.predict(input_data))
if __name__ == '__main__':
    pytest.main([__file__])