from caffe2.python import scope
from caffe2.python.model_helper import ModelHelper

class Seq2SeqModelHelper(ModelHelper):

    def __init__(self, init_params=True, **kwargs):
        if False:
            while True:
                i = 10
        arg_scope = {'use_cudnn': kwargs.pop('use_cudnn', True), 'cudnn_exhaustive_search': kwargs.pop('cudnn_exhaustive_search', False), 'order': 'NHWC'}
        if kwargs.get('ws_nbytes_limit', None):
            arg_scope['ws_nbytes_limit'] = kwargs.pop('ws_nbytes_limit')
        super().__init__(init_params=init_params, arg_scope=arg_scope, **kwargs)
        self.non_trainable_params = []

    def AddParam(self, name, init=None, init_value=None, trainable=True):
        if False:
            for i in range(10):
                print('nop')
        "Adds a parameter to the model's net and it's initializer if needed\n\n        Args:\n            init: a tuple (<initialization_op_name>, <initialization_op_kwargs>)\n            init_value: int, float or str. Can be used instead of `init` as a\n                simple constant initializer\n            trainable: bool, whether to compute gradient for this param or not\n        "
        if init_value is not None:
            assert init is None
            assert type(init_value) in [int, float, str]
            init = ('ConstantFill', dict(shape=[1], value=init_value))
        if self.init_params:
            param = self.param_init_net.__getattr__(init[0])([], name, **init[1])
        else:
            param = self.net.AddExternalInput(name)
        if trainable:
            self.params.append(param)
        else:
            self.non_trainable_params.append(param)
        return param

    def GetNonTrainableParams(self, namescope=None):
        if False:
            while True:
                i = 10
        '\n        Returns the params in current namescope\n        '
        if namescope is None:
            namescope = scope.CurrentNameScope()
        elif not namescope.endswith(scope._NAMESCOPE_SEPARATOR):
            namescope += scope._NAMESCOPE_SEPARATOR
        if namescope == '':
            return self.non_trainable_params[:]
        else:
            return [p for p in self.non_trainable_params if p.GetNameScope() == namescope]

    def GetAllParams(self, namescope=None):
        if False:
            i = 10
            return i + 15
        return self.GetParams(namescope) + self.GetComputedParams(namescope) + self.GetNonTrainableParams(namescope)