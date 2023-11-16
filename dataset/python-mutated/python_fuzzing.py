"""Helper class for TF Python fuzzing."""
import atheris
import tensorflow as tf
_MIN_INT = -10000
_MAX_INT = 10000
_MIN_FLOAT = -10000.0
_MAX_FLOAT = 10000.0
_MIN_LENGTH = 0
_MAX_LENGTH = 10000
_MIN_SIZE = 0
_MAX_SIZE = 8
_TF_DTYPES = [tf.half, tf.float16, tf.float32, tf.float64, tf.bfloat16, tf.complex64, tf.complex128, tf.int8, tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int16, tf.int32, tf.int64, tf.bool, tf.string, tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32, tf.resource, tf.variant]
_TF_RANDOM_DTYPES = [tf.float16, tf.float32, tf.float64, tf.int32, tf.int64]

class FuzzingHelper(object):
    """FuzzingHelper makes handling FuzzedDataProvider easier with TensorFlow Python fuzzing."""

    def __init__(self, input_bytes):
        if False:
            while True:
                i = 10
        'FuzzingHelper initializer.\n\n    Args:\n      input_bytes: Input randomized bytes used to create a FuzzedDataProvider.\n    '
        self.fdp = atheris.FuzzedDataProvider(input_bytes)

    def get_bool(self):
        if False:
            for i in range(10):
                print('nop')
        'Consume a bool.\n\n    Returns:\n      Consumed a bool based on input bytes and constraints.\n    '
        return self.fdp.ConsumeBool()

    def get_int(self, min_int=_MIN_INT, max_int=_MAX_INT):
        if False:
            while True:
                i = 10
        'Consume a signed integer with given constraints.\n\n    Args:\n      min_int: Minimum allowed integer.\n      max_int: Maximum allowed integer.\n\n    Returns:\n      Consumed integer based on input bytes and constraints.\n    '
        return self.fdp.ConsumeIntInRange(min_int, max_int)

    def get_float(self, min_float=_MIN_FLOAT, max_float=_MAX_FLOAT):
        if False:
            i = 10
            return i + 15
        'Consume a float with given constraints.\n\n    Args:\n      min_float: Minimum allowed float.\n      max_float: Maximum allowed float.\n\n    Returns:\n      Consumed float based on input bytes and constraints.\n    '
        return self.fdp.ConsumeFloatInRange(min_float, max_float)

    def get_int_list(self, min_length=_MIN_LENGTH, max_length=_MAX_LENGTH, min_int=_MIN_INT, max_int=_MAX_INT):
        if False:
            i = 10
            return i + 15
        'Consume a signed integer list with given constraints.\n\n    Args:\n      min_length: The minimum length of the list.\n      max_length: The maximum length of the list.\n      min_int: Minimum allowed integer.\n      max_int: Maximum allowed integer.\n\n    Returns:\n      Consumed integer list based on input bytes and constraints.\n    '
        length = self.get_int(min_length, max_length)
        return self.fdp.ConsumeIntListInRange(length, min_int, max_int)

    def get_float_list(self, min_length=_MIN_LENGTH, max_length=_MAX_LENGTH):
        if False:
            return 10
        'Consume a float list with given constraints.\n\n    Args:\n      min_length: The minimum length of the list.\n      max_length: The maximum length of the list.\n\n    Returns:\n      Consumed integer list based on input bytes and constraints.\n    '
        length = self.get_int(min_length, max_length)
        return self.fdp.ConsumeFloatListInRange(length, _MIN_FLOAT, _MAX_FLOAT)

    def get_int_or_float_list(self, min_length=_MIN_LENGTH, max_length=_MAX_LENGTH):
        if False:
            print('Hello World!')
        'Consume a signed integer or float list with given constraints based on a consumed bool.\n\n    Args:\n      min_length: The minimum length of the list.\n      max_length: The maximum length of the list.\n\n    Returns:\n      Consumed integer or float list based on input bytes and constraints.\n    '
        if self.get_bool():
            return self.get_int_list(min_length, max_length)
        else:
            return self.get_float_list(min_length, max_length)

    def get_tf_dtype(self, allowed_set=None):
        if False:
            print('Hello World!')
        'Return a random tensorflow dtype.\n\n    Args:\n      allowed_set: An allowlisted set of dtypes to choose from instead of all of\n      them.\n\n    Returns:\n      A random type from the list containing all TensorFlow types.\n    '
        if allowed_set:
            index = self.get_int(0, len(allowed_set) - 1)
            if allowed_set[index] not in _TF_DTYPES:
                raise tf.errors.InvalidArgumentError(None, None, 'Given dtype {} is not accepted.'.format(allowed_set[index]))
            return allowed_set[index]
        else:
            index = self.get_int(0, len(_TF_DTYPES) - 1)
            return _TF_DTYPES[index]

    def get_string(self, byte_count=_MAX_INT):
        if False:
            for i in range(10):
                print('nop')
        'Consume a string with given constraints based on a consumed bool.\n\n    Args:\n      byte_count: Byte count that defaults to _MAX_INT.\n\n    Returns:\n      Consumed string based on input bytes and constraints.\n    '
        return self.fdp.ConsumeString(byte_count)

    def get_random_numeric_tensor(self, dtype=None, min_size=_MIN_SIZE, max_size=_MAX_SIZE, min_val=_MIN_INT, max_val=_MAX_INT):
        if False:
            print('Hello World!')
        'Return a tensor of random shape and values.\n\n    Generated tensors are capped at dimension sizes of 8, as 2^32 bytes of\n    requested memory crashes the fuzzer (see b/34190148).\n    Returns only type that tf.random.uniform can generate. If you need a\n    different type, consider using tf.cast.\n\n    Args:\n      dtype: Type of tensor, must of one of the following types: float16,\n        float32, float64, int32, or int64\n      min_size: Minimum size of returned tensor\n      max_size: Maximum size of returned tensor\n      min_val: Minimum value in returned tensor\n      max_val: Maximum value in returned tensor\n\n    Returns:\n      Tensor of random shape filled with uniformly random numeric values.\n    '
        if max_size > 8:
            raise tf.errors.InvalidArgumentError(None, None, 'Given size of {} will result in an OOM error'.format(max_size))
        seed = self.get_int()
        shape = self.get_int_list(min_length=min_size, max_length=max_size, min_int=min_size, max_int=max_size)
        if dtype is None:
            dtype = self.get_tf_dtype(allowed_set=_TF_RANDOM_DTYPES)
        elif dtype not in _TF_RANDOM_DTYPES:
            raise tf.errors.InvalidArgumentError(None, None, 'Given dtype {} is not accepted in get_random_numeric_tensor'.format(dtype))
        return tf.random.uniform(shape=shape, minval=min_val, maxval=max_val, dtype=dtype, seed=seed)