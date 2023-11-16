from caffe2.python.core import DataType, BlobReference, ScopedBlobReference
from caffe2.python.modeling.parameter_info import ParameterInfo

class Initializer:
    """
    This class abstracts out parameter creation. One can come up with a new
    Initializer in order to implement more complex parameter initialization logic
    """

    def __init__(self, operator_name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.operator_name = operator_name
        self.operator_kwargs = kwargs

    def update(self, operator_name, kwargs):
        if False:
            return 10
        if self.operator_name is not None:
            raise Exception('Operator name overwrites are not allowed')
        self.operator_name = operator_name
        self.operator_kwargs = kwargs

    def create_param(self, param_name, init_net, shape):
        if False:
            i = 10
            return i + 15
        param = init_net.__getattr__(self.operator_name)([], param_name, shape=shape, **self.operator_kwargs)
        return ParameterInfo(param_id=None, param=param, shape=shape)

class ExternalInitializer:
    """
    This class is used in cases when the parameter should not be initialized by
    the initializer, but rather provided in the workspace when param_init_net is
    executed.

    Current version is not doing any real sanity checks to the parameter.
    """

    def create_param(self, param_name, init_net, shape):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(param_name, BlobReference):
            param = BlobReference(str(param_name), init_net)
        elif isinstance(param_name, str):
            param = ScopedBlobReference(param_name, init_net)
        else:
            raise TypeError('Unsupported type for param_name')
        return ParameterInfo(param_id=None, param=param, shape=shape)

class PseudoFP16Initializer(Initializer):
    """
    Used in cases when the parameter should be used at half (16-bit) precision
    for compute purposes (i.e. on the forward and backward pass) but
    needs to be stored and optimized at single (32-bit) precision so tiny
    gradients with small learning rates don't underflow FP16 precision.
    A 32-bit copy of the 16-bit blob is stored in the ParameterInfo.
    This is helpful for mixed-precision training, see
    https://arxiv.org/abs/1710.03740 for details.
    """

    def update(self, operator_name, kwargs):
        if False:
            while True:
                i = 10
        if self.operator_name is not None:
            raise Exception('Operator name overwrites are not allowed')
        self.operator_name = operator_name
        self.operator_kwargs = kwargs

    def create_param(self, param_name, init_net, shape):
        if False:
            print('Hello World!')
        param_fp32 = init_net.__getattr__(self.operator_name)([], param_name + '_fp32', shape=shape, **self.operator_kwargs)
        param = init_net.FloatToHalf(param_fp32, param_name)
        return ParameterInfo(param_id=None, param=param, shape=shape, blob_copy={DataType.FLOAT: param_fp32})

class ReversePseudoFP16Initializer(Initializer):
    """
    Like PseudoFP16Initializer above, except the primary blob is taken to
    be the 32-bit precision parameter, and the 16-bit version of the blob
    is stored in blob_copy instead.
    """

    def update(self, operator_name, kwargs):
        if False:
            while True:
                i = 10
        if self.operator_name is not None:
            raise Exception('Operator name overwrites are not allowed')
        self.operator_name = operator_name
        self.operator_kwargs = kwargs

    def create_param(self, param_name, init_net, shape):
        if False:
            for i in range(10):
                print('nop')
        param_fp32 = init_net.__getattr__(self.operator_name)([], param_name, shape=shape, **self.operator_kwargs)
        param_fp16 = init_net.FloatToHalf(param_fp32, param_name + '_fp16')
        return ParameterInfo(param_id=None, param=param_fp32, shape=shape, blob_copy={DataType.FLOAT16: param_fp16})

def update_initializer(initializer_class, operator_name_and_kwargs, default_operator_name_and_kwargs):
    if False:
        i = 10
        return i + 15
    '\n    A helper function to convert from operator_name_and_kwargs to new\n    object of type initializer_class. This function serves two purposes:\n\n    1. Support for custom initialization operators being passed in\n    2. Allow user to specify a custom Initializer without overwriting\n       default operators used for initialization\n\n    If initializer_class is None, creates a default initializer using\n    the Initializer class and operator_name_and_kwargs provided\n\n    If operator_name_and_kwargs is None, uses default_operator_name_and_kwargs\n\n    returns an instantiated Initializer object\n    '

    def get_initializer_args():
        if False:
            i = 10
            return i + 15
        return operator_name_and_kwargs or default_operator_name_and_kwargs
    if initializer_class is not None:
        init = initializer_class(get_initializer_args()[0], **get_initializer_args()[1])
    else:
        init = Initializer(get_initializer_args()[0], **get_initializer_args()[1])
    return init