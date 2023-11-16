import os
import sys
import gc
from tempfile import NamedTemporaryFile
from pyspark.broadcast import Broadcast
from pyspark.broadcast import _from_id
from bigdl.dllib.nn.layer import Model
from bigdl.dllib.utils.log4Error import *

def _from_id_and_type(bid, bigdl_type):
    if False:
        return 10
    result = _from_id(bid)
    return ModelBroadcast(path=result._path, bigdl_type=bigdl_type)

def broadcast_model(sc, layer):
    if False:
        print('Hello World!')
    return ModelBroadcast(sc, layer, sc._pickled_broadcast_vars)

class ModelBroadcast(Broadcast):

    def __init__(self, sc=None, layer=None, pickle_registry=None, path=None, bigdl_type='float'):
        if False:
            return 10
        '\n        Should not be called directly by users -- use L{SparkContext.broadcast()}\n        instead.\n        '
        if layer is not None:
            self.bigdl_type = layer.bigdl_type
        else:
            self.bigdl_type = bigdl_type
        super(ModelBroadcast, self).__init__(sc, layer, pickle_registry, path)

    def dump(self, value, f):
        if False:
            while True:
                i = 10
        try:
            value.saveModel(f.name, over_write=True)
        except Exception as e:
            msg = 'Could not serialize broadcast: %s' % e.__class__.__name__
            if not self.sc.version.startswith('2.1'):
                from pyspark.cloudpickle import print_exec
            else:
                from pyspark.util import print_exec
            print_exec(sys.stderr)
            invalidInputError(False, msg)
        f.close()
        return f.name

    def _load(self, path):
        if False:
            return 10
        return Model.loadModel(path, bigdl_type=self.bigdl_type)

    @property
    def value(self):
        if False:
            return 10
        ' Return the broadcasted value\n        '
        if not hasattr(self, '_value') and self._path is not None:
            self._value = self._load(self._path)
        return self._value

    def __reduce__(self):
        if False:
            while True:
                i = 10
        if self._jbroadcast is None:
            invalidInputError(False, 'Broadcast can only be serialized in driver')
        self._pickle_registry.add(self)
        return (_from_id_and_type, (self._jbroadcast.id(), self.bigdl_type))