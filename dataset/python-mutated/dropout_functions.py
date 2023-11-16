import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
@with_unsupported_dtypes({'2.1.0 and below': ('float16',)}, 'torch')
@with_unsupported_dtypes({'2.5.2 and below': ('float16', 'bfloat16')}, 'paddle')
def alpha_dropout(input, p=0.5, training=False, inplace=False):
    if False:
        i = 10
        return i + 15
    if p == 0.0 or not training or input.shape == () or (input.shape == (0,)):
        return input
    neg_saturation = ivy.log1p(ivy.exp(-ivy.square(input)))
    mask = ivy.where(ivy.random_uniform(shape=input.shape, device=ivy.dev(input)) < p, 0.0, 1.0)
    if inplace:
        ivy.inplace_update(input, mask * input + (1 - mask) * neg_saturation)
        ivy.inplace_update(input, input / ivy.sqrt(1 - p / (1 - p + 1e-05)))
        return input
    else:
        masked = mask * input + (1 - mask) * neg_saturation
        return masked / ivy.sqrt(1 - p / (1 - p + 1e-05))

@to_ivy_arrays_and_back
@with_unsupported_dtypes({'2.1.0 and below': ('float16',)}, 'torch')
def dropout(input, p=0.5, training=True, inplace=False):
    if False:
        for i in range(10):
            print('nop')
    return ivy.dropout(input, p, training=training)

@to_ivy_arrays_and_back
@with_unsupported_dtypes({'2.1.0 and below': ('float16',)}, 'torch')
def dropout1d(input, p=0.5, training=True, inplace=False):
    if False:
        return 10
    if inplace:
        return ivy.dropout1d(input, p, training=training, data_format='NCW', out=input)
    return ivy.dropout1d(input, p, training=training, data_format='NCW')

@to_ivy_arrays_and_back
@with_unsupported_dtypes({'2.1.0 and below': ('float16',)}, 'torch')
def dropout2d(input, p=0.5, training=True, inplace=False):
    if False:
        print('Hello World!')
    if input.ndim < 2:
        raise ValueError('Feature dropout requires at least 2 dimensions in the input')
    ret = ivy.dropout2d(input, p, training=training, data_format='NCHW')
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret

@to_ivy_arrays_and_back
@with_unsupported_dtypes({'2.1.0 and below': ('float16', 'bfloat16')}, 'torch')
def dropout3d(input, p=0.5, training=True, inplace=False):
    if False:
        for i in range(10):
            print('nop')
    if inplace:
        return ivy.dropout3d(input, p, training=training, data_format='NDHWC', out=input)
    return ivy.dropout3d(input, p, training=training, data_format='NDHWC')