from typing import List, Union, Sequence, Any
from functools import reduce
import numpy as np
from deeplake.core.linked_sample import LinkedSample
from deeplake.util.exceptions import TensorDtypeMismatchError
from deeplake.core.sample import Sample
import deeplake

def _get_bigger_dtype(d1, d2):
    if False:
        for i in range(10):
            print('nop')
    if np.can_cast(d1, d2):
        if np.can_cast(d2, d1):
            return d1
        else:
            return d2
    elif np.can_cast(d2, d1):
        return d2
    else:
        return np.object

def get_dtype(val: Union[np.ndarray, Sequence, Sample]) -> np.dtype:
    if False:
        while True:
            i = 10
    'Get the dtype of a non-uniform mixed dtype sequence of samples.'
    if hasattr(val, 'dtype'):
        return np.dtype(val.dtype)
    elif isinstance(val, int):
        return np.array(0).dtype
    elif isinstance(val, float):
        return np.array(0.0).dtype
    elif isinstance(val, str):
        return np.array('').dtype
    elif isinstance(val, bool):
        return np.dtype(bool)
    elif isinstance(val, Sequence):
        return reduce(_get_bigger_dtype, map(get_dtype, val))
    else:
        raise TypeError(f'Cannot infer numpy dtype for {val}')

def get_htype(val: Union[np.ndarray, Sequence, Sample]) -> str:
    if False:
        i = 10
        return i + 15
    'Get the htype of a non-uniform mixed dtype sequence of samples.'
    if isinstance(val, deeplake.core.tensor.Tensor):
        return val.meta.htype
    if hasattr(val, 'shape'):
        return 'generic'
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (Sample, LinkedSample)):
        return 'generic'
    types = set(map(type, val))
    if dict in types:
        return 'json'
    if types == set((str,)):
        return 'text'
    if object in [np.array(x).dtype if not isinstance(x, np.ndarray) else x.dtype for x in val if x is not None]:
        return 'json'
    return 'generic'

def get_empty_text_like_sample(htype: str):
    if False:
        for i in range(10):
            print('nop')
    "Get an empty sample of the given htype.\n\n    Args:\n        htype: htype of the sample.\n\n    Returns:\n        Empty sample.\n\n    Raises:\n        ValueError: if htype is not one of 'text', 'json', and 'list'.\n    "
    if htype == 'text':
        return ''
    elif htype == 'json':
        return {}
    elif htype == 'list':
        return []
    else:
        raise ValueError(f"This method should only be used for htypes 'text', 'json' and 'list'. Got {htype}.")

def intelligent_cast(sample: Any, dtype: Union[np.dtype, str], htype: str) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(sample, Sample):
        sample = sample.array
    if hasattr(sample, 'dtype') and sample.dtype == dtype:
        return sample
    err_dtype = get_incompatible_dtype(sample, dtype)
    if err_dtype:
        raise TensorDtypeMismatchError(dtype, err_dtype, htype)
    if hasattr(sample, 'astype'):
        return sample.astype(dtype)
    return np.array(sample, dtype=dtype)

def get_incompatible_dtype(samples: Union[np.ndarray, Sequence], dtype: Union[str, np.dtype]):
    if False:
        while True:
            i = 10
    'Check if items in a non-uniform mixed dtype sequence of samples can be safely cast to the given dtype.\n\n    Args:\n        samples: Sequence of samples\n        dtype: dtype to which samples have to be cast\n\n    Returns:\n        None if all samples are compatible. If not, the dtype of the offending item is returned.\n\n    Raises:\n        TypeError: if samples is of unexepcted type.\n    '
    if isinstance(samples, np.ndarray):
        if samples.size == 0:
            return None
        elif samples.size == 1:
            samples = samples.reshape(1).tolist()[0]
    if isinstance(samples, (int, float, bool)) or hasattr(samples, 'dtype'):
        return None if np.can_cast(samples, dtype) else getattr(samples, 'dtype', np.array(samples).dtype)
    elif isinstance(samples, str):
        return None if dtype == np.dtype(str) else np.dtype(str)
    elif isinstance(samples, Sequence):
        for dt in map(lambda x: get_incompatible_dtype(x, dtype), samples):
            if dt:
                return dt
        return None
    else:
        raise TypeError(f'Unexpected object {samples}. Expected np.ndarray, int, float, bool, str or Sequence.')