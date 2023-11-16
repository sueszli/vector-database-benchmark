import sys
from bigdl.dllib.utils.common import JavaValue
if sys.version >= '3':
    long = int
    unicode = str

class InitializationMethod(JavaValue):
    """
    Initialization method to initialize bias and weight.
    The init method will be called in Module.reset()
    """

class Zeros(InitializationMethod):
    """
    Initializer that generates tensors with zeros.
    """

    def __init__(self, bigdl_type='float'):
        if False:
            return 10
        JavaValue.__init__(self, None, bigdl_type)

class Ones(InitializationMethod):
    """
    Initializer that generates tensors with ones.
    """

    def __init__(self, bigdl_type='float'):
        if False:
            i = 10
            return i + 15
        JavaValue.__init__(self, None, bigdl_type)

class RandomUniform(InitializationMethod):
    """
     Initializer that generates tensors with a uniform distribution.
     It draws samples from a uniform distribution within [lower, upper]
     If lower and upper is not specified, it draws samples form a
     uniform distribution within [-limit, limit] where "limit" is "1/sqrt(fan_in)"
    """

    def __init__(self, upper=None, lower=None, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        if upper is not None and lower is not None:
            upper = upper + 0.0
            lower = lower + 0.0
            JavaValue.__init__(self, None, bigdl_type, upper, lower)
        else:
            JavaValue.__init__(self, None, bigdl_type)

class RandomNormal(InitializationMethod):
    """
     Initializer that generates tensors with a normal distribution.
    """

    def __init__(self, mean, stdv, bigdl_type='float'):
        if False:
            while True:
                i = 10
        mean = mean + 0.0
        stdv = stdv + 0.0
        JavaValue.__init__(self, None, bigdl_type, mean, stdv)

class ConstInitMethod(InitializationMethod):
    """
    Initializer that generates tensors with certain constant double.
    """

    def __init__(self, value, bigdl_type='float'):
        if False:
            i = 10
            return i + 15
        value = value + 0.0
        JavaValue.__init__(self, None, bigdl_type, value)

class Xavier(InitializationMethod):
    """
    Xavier Initializer. See http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """

    def __init__(self, bigdl_type='float'):
        if False:
            while True:
                i = 10
        JavaValue.__init__(self, None, bigdl_type)

class MsraFiller(InitializationMethod):
    """
    MsraFiller Initializer.
    See https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_
    2015_paper.pdf
    """

    def __init__(self, varianceNormAverage=True, bigdl_type='float'):
        if False:
            return 10
        JavaValue.__init__(self, None, bigdl_type, varianceNormAverage)

class BilinearFiller(InitializationMethod):
    """
    Initialize the weight with coefficients for bilinear interpolation.

    A common use case is with the DeconvolutionLayer acting as upsampling.
    The variable tensor passed in the init function should have 5 dimensions
    of format [nGroup, nInput, nOutput, kH, kW], and kH should be equal to kW
    """

    def __init__(self, bigdl_type='float'):
        if False:
            for i in range(10):
                print('nop')
        JavaValue.__init__(self, None, bigdl_type)