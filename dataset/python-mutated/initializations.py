import numpy as np
'\nReferences:\nhttp://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf\n\n'

def normal(shape, scale=0.5):
    if False:
        i = 10
        return i + 15
    return np.random.normal(size=shape, scale=scale)

def uniform(shape, scale=0.5):
    if False:
        return 10
    return np.random.uniform(size=shape, low=-scale, high=scale)

def zero(shape, **kwargs):
    if False:
        while True:
            i = 10
    return np.zeros(shape)

def one(shape, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return np.ones(shape)

def orthogonal(shape, scale=0.5):
    if False:
        return 10
    flat_shape = (shape[0], np.prod(shape[1:]))
    array = np.random.normal(size=flat_shape)
    (u, _, v) = np.linalg.svd(array, full_matrices=False)
    array = u if u.shape == flat_shape else v
    return np.reshape(array * scale, shape)

def _glorot_fan(shape):
    if False:
        return 10
    assert len(shape) >= 2
    if len(shape) == 4:
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        (fan_in, fan_out) = shape[:2]
    return (float(fan_in), float(fan_out))

def glorot_normal(shape, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    (fan_in, fan_out) = _glorot_fan(shape)
    s = np.sqrt(2.0 / (fan_in + fan_out))
    return normal(shape, s)

def glorot_uniform(shape, **kwargs):
    if False:
        while True:
            i = 10
    (fan_in, fan_out) = _glorot_fan(shape)
    s = np.sqrt(6.0 / (fan_in + fan_out))
    return uniform(shape, s)

def he_normal(shape, **kwargs):
    if False:
        return 10
    (fan_in, fan_out) = _glorot_fan(shape)
    s = np.sqrt(2.0 / fan_in)
    return normal(shape, s)

def he_uniform(shape, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    (fan_in, fan_out) = _glorot_fan(shape)
    s = np.sqrt(6.0 / fan_in)
    return uniform(shape, s)

def get_initializer(name):
    if False:
        for i in range(10):
            print('nop')
    'Returns initialization function by the name.'
    try:
        return globals()[name]
    except Exception:
        raise ValueError('Invalid initialization function.')