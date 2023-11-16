import sys
from bigdl.dllib.nn.criterion import Criterion
from pyspark.serializers import CloudPickleSerializer
from importlib.util import find_spec
from bigdl.dllib.utils.log4Error import invalidInputError
from deprecated import deprecated
if sys.version < '3.7':
    print('WARN: detect python < 3.7, if you meet zlib not available ' + 'exception on yarn, please update your python to 3.7')
if find_spec('jep') is None:
    invalidInputError(False, 'jep not found, please install jep first.')

@deprecated(version='2.3.0', reason='Please use spark engine and ray engine.')
class TorchLoss(Criterion):
    """
    TorchLoss wraps a loss function for distributed inference or training.
    This TorchLoss should be used with TorchModel.
    """

    def __init__(self, criterion_bytes, bigdl_type='float'):
        if False:
            i = 10
            return i + 15
        '\n        :param bigdl_type:\n        '
        super(TorchLoss, self).__init__(None, bigdl_type, criterion_bytes)

    @staticmethod
    def from_pytorch(criterion):
        if False:
            return 10
        bys = CloudPickleSerializer.dumps(CloudPickleSerializer, criterion)
        net = TorchLoss(bys)
        return net