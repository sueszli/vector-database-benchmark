import numpy as np
from . import core
from .data_feeder import DataToLoDTensorConverter
__all__ = []

def create_lod_tensor(data, recursive_seq_lens, place):
    if False:
        while True:
            i = 10
    '\n    Create a LoDTensor from a numpy array, list or existing LoDTensor.\n\n    The implementation is as follows:\n\n    1. Check whether the length-based LoD, i.e., :code:`recursive_seq_lens`\n       is valid.\n\n    2. Convert :code:`recursive_seq_lens` to a offset-based LoD.\n\n    3. Based on :code:`place` , copy the :code:`data` from a numpy array, list\n       or existing LoDTensor to CPU or GPU device.\n\n    4. Set offset-based LoD to the output LoDTensor.\n\n    Suppose we want to create a LoDTensor to hold data for word sequences,\n    where each word is represented by an integer. If we want to create\n    a LoDTensor to represent two sentences, one of 2 words, and one of 3 words.\n\n    Then :code:`data` would be a numpy array of integers with shape (5, 1).\n    :code:`recursive_seq_lens` would be [[2, 3]], indicating the word number\n    in each sentence. This length-based :code:`recursive_seq_lens` [[2, 3]]\n    would be converted to offset-based LoD [[0, 2, 5]] inside the function\n    call.\n\n\n    Args:\n        data (numpy.ndarray|list|LoDTensor): a numpy array, a list or ad LoDTensor\n                holding the data to be copied.\n        recursive_seq_lens (list[list[int]]): a list of lists indicating the\n                length-based LoD info.\n        place (CPUPlace|CUDAPlace): CPU or GPU place indicating where the data\n                in the created LoDTensor will be stored.\n\n    Returns:\n         A LoDTensor with tensor data and recursive_seq_lens info.\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> import paddle.base as base\n            >>> import numpy as np\n\n            >>> t = base.create_lod_tensor(np.ndarray([5, 30]), [[2, 3]], base.CPUPlace())\n    '
    if isinstance(data, core.LoDTensor):
        return create_lod_tensor(np.array(data), recursive_seq_lens, place)
    elif isinstance(data, list):
        converter = DataToLoDTensorConverter(place=place, lod_level=len(recursive_seq_lens), shape=[], dtype=core.VarDesc.VarType.FP32)
        new_recursive_seq_lens = []
        for seq in data:
            new_recursive_seq_lens.append(len(seq))
            converter.feed(seq)
        assert [new_recursive_seq_lens] == recursive_seq_lens, 'data and recursive_seq_lens do not match'
        arr = np.array(converter.data)
        arr = arr.reshape(arr.shape + (1,))
        tensor = core.LoDTensor()
        tensor.set(arr, place)
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        return tensor
    elif isinstance(data, np.ndarray):
        tensor = core.LoDTensor()
        tensor.set(data, place)
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        assert tensor.has_valid_recursive_sequence_lengths(), 'the provided lod info is invalid'
        return tensor
    else:
        raise TypeError('data should be either a LoDTensor, a Numpy array or a list')

def create_random_int_lodtensor(recursive_seq_lens, base_shape, place, low, high):
    if False:
        print('Hello World!')
    '\n        :api_attr: Static Graph\n\n    Create a LoDTensor containing random integers.\n\n    The implementation is as follows:\n\n    1. Obtain the shape of output LoDTensor based on :code:`recursive_seq_lens`\n       and :code:`base_shape` . The first dimension of the shape is the total\n       length of sequences, while the other dimensions are the same as\n       :code:`base_shape` .\n\n    2. Create a numpy array of random integers, and parse the created numpy\n       array as parameter :code:`data` of :ref:`api_paddle_base_create_lod_tensor` to\n       create the output LoDTensor.\n\n    Suppose we want to create a LoDTensor to hold data for 2 sequences, where\n    the dimension of the sequences are [2, 30] and [3, 30] respectively.\n    The :code:`recursive_seq_lens` would be [[2, 3]], and :code:`base_shape`\n    would be [30] (the other dimensions excluding the sequence length).\n    Therefore, the shape of the output LoDTensor would be [5, 30], where\n    the first dimension 5 is the total lengths of the sequences, and the\n    other dimensions are :code:`base_shape`.\n\n    Args:\n        recursive_seq_lens (list[list[int]]): a list of lists indicating the\n                length-based LoD info.\n        base_shape (list[int]): the shape of the output LoDTensor excluding\n                the first dimension.\n        place (CPUPlace|CUDAPlace): CPU or GPU place indicating where\n                the data in the created LoDTensor will be stored.\n        low (int): the lower bound of the random integers.\n        high (int): the upper bound of the random integers.\n\n    Returns:\n        A LoDTensor with tensor data and recursive_seq_lens info, whose data\n        is inside [low, high].\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> import paddle.base as base\n\n            >>> t = base.create_random_int_lodtensor(recursive_seq_lens=[[2, 3]],\n            ...         base_shape=[30], place=base.CPUPlace(), low=0, high=10)\n            >>> print(t.shape())\n            [5, 30]\n    '
    assert isinstance(base_shape, list), 'base_shape should be a list'
    overall_shape = [sum(recursive_seq_lens[-1])] + base_shape
    data = np.random.random_integers(low, high, overall_shape).astype('int64')
    return create_lod_tensor(data, recursive_seq_lens, place)