import importlib
import os
import sys
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.nn.layer import Model as BModel
from bigdl.dllib.net.graph_net import GraphNet
from bigdl.dllib.utils.log4Error import *
if sys.version >= '3':
    long = int
    unicode = str

class JavaToPython:

    def __init__(self, jvalue, bigdl_type='float'):
        if False:
            i = 10
            return i + 15
        self.jvaule = jvalue
        self.jfullname = callZooFunc(bigdl_type, 'getRealClassNameOfJValue', jvalue)

    def get_python_class(self):
        if False:
            return 10
        '\n        Redirect the jvalue to the proper python class.\n        :param jvalue: Java object create by Py4j\n        :return: A proper Python wrapper which would be a Model, Sequential...\n        '
        jpackage_name = '.'.join(self.jfullname.split('.')[:-1])
        pclass_name = self._get_py_name(self.jfullname.split('.')[-1])
        if self.jfullname in ('com.intel.analytics.bigdl.dllib.keras.Sequential', 'com.intel.analytics.bigdl.dllib.keras.Model'):
            base_module = self._load_ppackage_by_jpackage(self.jfullname)
        else:
            base_module = self._load_ppackage_by_jpackage(jpackage_name)
        if pclass_name in dir(base_module):
            pclass = getattr(base_module, pclass_name)
            invalidInputError('from_jvalue' in dir(pclass), 'pclass: {} should implement from_jvalue method'.format(pclass))
            return pclass
        invalidInputError(False, 'No proper python class for: {}'.format(self.jfullname))

    def _get_py_name(self, jclass_name):
        if False:
            print('Hello World!')
        if jclass_name == 'Model':
            return 'Model'
        elif jclass_name == 'Sequential':
            return 'Sequential'
        else:
            invalidInputError(False, 'Not supported type: {}'.format(jclass_name))

    def _load_ppackage_by_jpackage(self, jpackage_name):
        if False:
            return 10
        if jpackage_name in ('com.intel.analytics.bigdl.dllib.keras.Model', 'com.intel.analytics.bigdl.dllib.keras.Sequential'):
            return importlib.import_module('bigdl.dllib.keras.models')
        invalidInputError(False, 'Not supported package: {}'.format(jpackage_name))

class Net:

    @staticmethod
    def from_jvalue(jvalue, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        pclass = JavaToPython(jvalue).get_python_class()
        return getattr(pclass, 'from_jvalue')(jvalue, bigdl_type)

    @staticmethod
    def load_bigdl(model_path, weight_path=None, bigdl_type='float'):
        if False:
            while True:
                i = 10
        '\n        Load a pre-trained BigDL model.\n\n        :param model_path: The path to the pre-trained model.\n        :param weight_path: The path to the weights of the pre-trained model. Default is None.\n        :return: A pre-trained model.\n        '
        jmodel = callZooFunc(bigdl_type, 'netLoadBigDL', model_path, weight_path)
        return GraphNet.from_jvalue(jmodel)

    @staticmethod
    def load(model_path, weight_path=None, bigdl_type='float'):
        if False:
            print('Hello World!')
        "\n        Load an existing BigDL model defined in Keras-style(with weights).\n\n        :param model_path: The path to load the saved model.\n                          Local file system, HDFS and Amazon S3 are supported.\n                          HDFS path should be like 'hdfs://[host]:[port]/xxx'.\n                          Amazon S3 path should be like 's3a://bucket/xxx'.\n        :param weight_path: The path for pre-trained weights if any. Default is None.\n        :return: A BigDL model.\n        "
        jmodel = callZooFunc(bigdl_type, 'netLoad', model_path, weight_path)
        return Net.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_torch(path, bigdl_type='float'):
        if False:
            print('Hello World!')
        '\n        Load a pre-trained Torch model.\n\n        :param path: The path containing the pre-trained model.\n        :return: A pre-trained model.\n        '
        jmodel = callZooFunc(bigdl_type, 'netLoadTorch', path)
        return GraphNet.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_caffe(def_path, model_path, bigdl_type='float'):
        if False:
            print('Hello World!')
        '\n        Load a pre-trained Caffe model.\n\n        :param def_path: The path containing the caffe model definition.\n        :param model_path: The path containing the pre-trained caffe model.\n        :return: A pre-trained model.\n        '
        jmodel = callZooFunc(bigdl_type, 'netLoadCaffe', def_path, model_path)
        return GraphNet.from_jvalue(jmodel, bigdl_type)

    @staticmethod
    def load_keras(json_path=None, hdf5_path=None, by_name=False):
        if False:
            return 10
        '\n        Load a pre-trained Keras model.\n\n        :param json_path: The json path containing the keras model definition. Default is None.\n        :param hdf5_path: The HDF5 path containing the pre-trained keras model weights\n                        with or without the model architecture. Default is None.\n        :param by_name: by default the architecture should be unchanged.\n                        If set as True, only layers with the same name will be loaded.\n        :return: A BigDL model.\n        '
        return BModel.load_keras(json_path, hdf5_path, by_name)