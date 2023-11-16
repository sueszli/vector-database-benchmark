import sys
from bigdl.dllib.nn.layer import Model as BModel
from bigdl.dllib.feature.image import ImageSet
from bigdl.dllib.feature.text import TextSet
from bigdl.dllib.keras.base import ZooKerasLayer
from bigdl.dllib.keras.utils import *
from bigdl.dllib.nn.layer import Layer
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.utils.log4Error import *
if sys.version >= '3':
    long = int
    unicode = str

class GraphNet(BModel):

    def __init__(self, input, output, jvalue=None, bigdl_type='float', **kwargs):
        if False:
            return 10
        super(BModel, self).__init__(jvalue, to_list(input), to_list(output), bigdl_type, **kwargs)

    def predict(self, x, batch_per_thread=4, distributed=True):
        if False:
            print('Hello World!')
        '\n        Use a model to do prediction.\n\n        # Arguments\n        x: Prediction data. A Numpy array or RDD of Sample or ImageSet.\n        batch_per_thread:\n          The default value is 4.\n          When distributed is True,the total batch size is batch_per_thread * rdd.getNumPartitions.\n          When distributed is False the total batch size is batch_per_thread * numOfCores.\n        distributed: Boolean. Whether to do prediction in distributed mode or local mode.\n                     Default is True. In local mode, x must be a Numpy array.\n        '
        if isinstance(x, ImageSet) or isinstance(x, TextSet):
            results = callZooFunc(self.bigdl_type, 'zooPredict', self.value, x, batch_per_thread)
            return ImageSet(results) if isinstance(x, ImageSet) else TextSet(results)
        if distributed:
            if isinstance(x, np.ndarray):
                data_rdd = to_sample_rdd(x, np.zeros([x.shape[0]]))
            elif isinstance(x, RDD):
                data_rdd = x
            else:
                invalidInputError(False, 'Unsupported prediction data type: %s' % type(x))
            results = callZooFunc(self.bigdl_type, 'zooPredict', self.value, data_rdd, batch_per_thread)
            return results.map(lambda result: Layer.convert_output(result))
        elif isinstance(x, np.ndarray) or isinstance(x, list):
            results = callZooFunc(self.bigdl_type, 'zooPredict', self.value, self._to_jtensors(x), batch_per_thread)
            return [Layer.convert_output(result) for result in results]
        else:
            invalidInputError(False, 'Unsupported prediction data type: %s' % type(x))

    def flattened_layers(self, include_container=False):
        if False:
            while True:
                i = 10
        jlayers = callZooFunc(self.bigdl_type, 'getFlattenSubModules', self, include_container)
        layers = [Layer.of(jlayer) for jlayer in jlayers]
        return layers

    @property
    def layers(self):
        if False:
            for i in range(10):
                print('nop')
        jlayers = callZooFunc(self.bigdl_type, 'getSubModules', self)
        layers = [Layer.of(jlayer) for jlayer in jlayers]
        return layers

    @staticmethod
    def from_jvalue(jvalue, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a Python Model base on the given java value\n\n        :param jvalue: Java object create by Py4j\n        :return: A Python Model\n        '
        model = GraphNet([], [], jvalue=jvalue, bigdl_type=bigdl_type)
        model.value = jvalue
        return model

    def new_graph(self, outputs):
        if False:
            return 10
        '\n        Specify a list of nodes as output and return a new graph using the existing nodes\n\n        :param outputs: A list of nodes specified\n        :return: A graph model\n        '
        value = callZooFunc(self.bigdl_type, 'newGraph', self.value, outputs)
        return self.from_jvalue(value, self.bigdl_type)

    def freeze_up_to(self, names):
        if False:
            return 10
        '\n        Freeze the model from the bottom up to the layers specified by names (inclusive).\n        This is useful for finetuning a model\n\n        :param names: A list of module names to be Freezed\n        :return: current graph model\n        '
        callZooFunc(self.bigdl_type, 'freezeUpTo', self.value, names)

    def unfreeze(self, names=None):
        if False:
            i = 10
            return i + 15
        '\n        "unfreeze" module, i.e. make the module parameters(weight/bias, if exists)\n        to be trained(updated) in training process.\n        If \'names\' is a non-empty list, unfreeze layers that match given names\n\n        :param names: list of module names to be unFreezed. Default is None.\n        :return: current graph model\n        '
        callZooFunc(self.bigdl_type, 'unFreeze', self.value, names)

    def to_keras(self):
        if False:
            print('Hello World!')
        value = callZooFunc(self.bigdl_type, 'netToKeras', self.value)
        return ZooKerasLayer.of(value, self.bigdl_type)