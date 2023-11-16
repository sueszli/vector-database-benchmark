import numpy as np
from keras.api_export import keras_export

@keras_export(['keras.utils.pad_sequences', 'keras.preprocessing.sequence.pad_sequences'])
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0):
    if False:
        print('Hello World!')
    'Pads sequences to the same length.\n\n    This function transforms a list (of length `num_samples`)\n    of sequences (lists of integers)\n    into a 2D NumPy array of shape `(num_samples, num_timesteps)`.\n    `num_timesteps` is either the `maxlen` argument if provided,\n    or the length of the longest sequence in the list.\n\n    Sequences that are shorter than `num_timesteps`\n    are padded with `value` until they are `num_timesteps` long.\n\n    Sequences longer than `num_timesteps` are truncated\n    so that they fit the desired length.\n\n    The position where padding or truncation happens is determined by\n    the arguments `padding` and `truncating`, respectively.\n    Pre-padding or removing values from the beginning of the sequence is the\n    default.\n\n    >>> sequence = [[1], [2, 3], [4, 5, 6]]\n    >>> keras.utils.pad_sequences(sequence)\n    array([[0, 0, 1],\n           [0, 2, 3],\n           [4, 5, 6]], dtype=int32)\n\n    >>> keras.utils.pad_sequences(sequence, value=-1)\n    array([[-1, -1,  1],\n           [-1,  2,  3],\n           [ 4,  5,  6]], dtype=int32)\n\n    >>> keras.utils.pad_sequences(sequence, padding=\'post\')\n    array([[1, 0, 0],\n           [2, 3, 0],\n           [4, 5, 6]], dtype=int32)\n\n    >>> keras.utils.pad_sequences(sequence, maxlen=2)\n    array([[0, 1],\n           [2, 3],\n           [5, 6]], dtype=int32)\n\n    Args:\n        sequences: List of sequences (each sequence is a list of integers).\n        maxlen: Optional Int, maximum length of all sequences. If not provided,\n            sequences will be padded to the length of the longest individual\n            sequence.\n        dtype: (Optional, defaults to `"int32"`). Type of the output sequences.\n            To pad sequences with variable length strings, you can use `object`.\n        padding: String, "pre" or "post" (optional, defaults to `"pre"`):\n            pad either before or after each sequence.\n        truncating: String, "pre" or "post" (optional, defaults to `"pre"`):\n            remove values from sequences larger than\n            `maxlen`, either at the beginning or at the end of the sequences.\n        value: Float or String, padding value. (Optional, defaults to 0.)\n\n    Returns:\n        NumPy array with shape `(len(sequences), maxlen)`\n    '
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)
    lengths = []
    sample_shape = ()
    flag = True
    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(f'`sequences` must be a list of iterables. Found non-iterable: {str(x)}') from e
    if maxlen is None:
        maxlen = np.max(lengths)
    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and (not is_dtype_str):
        raise ValueError(f"`dtype` {dtype} is not compatible with `value`'s type: {type(value)}\nYou should set `dtype=object` for variable length strings.")
    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for (idx, s) in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(f'Shape of sample {trunc.shape[1:]} of sequence at position {idx} is different from expected shape {sample_shape}')
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x