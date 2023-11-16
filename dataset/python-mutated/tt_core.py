import numpy as np
'\nThe following methods are various utility methods for using the Tensor-Train\ndecomposition, or TT-decomposition introduced by I. V. Oseledets (2011) in his\npaper (http://epubs.siam.org/doi/abs/10.1137/090752286).\n\nBroadly speaking, these methods are used to replace fully connected layers in\nneural networks with Tensor-Train layers introduced by A. Novikov et. al. (2015)\nin their paper (http://arxiv.org/abs/1509.06569). More details about each of\nthe methods are provided in each respective docstring.\n'

def init_tt_cores(inp_sizes, out_sizes, tt_ranks, seed=1234):
    if False:
        print('Hello World!')
    '\n    Initialize randomized orthogonalized TT-cores.\n\n    This method should be used when a TT-layer is trained from scratch. The\n    sizes of each of the cores are specified by the inp_sizes and out_sizes, and\n    the respective tt_ranks will dictate the ranks of each of the cores. Note\n    that a larger set of tt_ranks will result in slower computation but will\n    result in more accurate approximations. The size of the ith core is:\n\n        tt_ranks[i] * inp_sizes[i] * out_sizes[i] * tt_ranks[i + 1].\n\n    Note that the following relationships of lengths of each input is expected:\n\n        len(inp_sizes) == len(out_sizes) == len(tt_ranks) - 1.\n\n    Args:\n        inp_sizes: list of the input dimensions of the respective cores\n        out_sizes: list of the output dimensions of the respective cores\n        tt_ranks: list of the ranks of the respective cores\n        seed: integer to seed the random number generator\n\n    Returns:\n        cores: One-dimensional list of cores concatentated along an axis\n    '
    np.random.seed(seed)
    assert len(inp_sizes) == len(out_sizes), 'The number of input dimensions (' + str(len(inp_sizes)) + ') must be equal to the number of output dimensions (' + str(len(out_sizes)) + ').'
    assert len(tt_ranks) == len(inp_sizes) + 1, 'The number of tt-ranks (' + str(len(tt_ranks)) + ') must be ' + 'one more than the number of input and output dims (' + str(len(out_sizes)) + ').'
    inp_sizes = np.array(inp_sizes)
    out_sizes = np.array(out_sizes)
    tt_ranks = np.array(tt_ranks)
    cores_len = np.sum(inp_sizes * out_sizes * tt_ranks[1:] * tt_ranks[:-1])
    cores = np.zeros(cores_len)
    cores_idx = 0
    rv = 1
    for i in range(inp_sizes.shape[0]):
        shape = [tt_ranks[i], inp_sizes[i], out_sizes[i], tt_ranks[i + 1]]
        tall_shape = (np.prod(shape[:3]), shape[3])
        curr_core = np.dot(rv, np.random.normal(0, 1, size=(shape[0], np.prod(shape[1:]))))
        curr_core = curr_core.reshape(tall_shape)
        if i < inp_sizes.shape[0] - 1:
            (curr_core, rv) = np.linalg.qr(curr_core)
        cores[cores_idx:cores_idx + curr_core.size] = curr_core.flatten()
        cores_idx += curr_core.size
    glarot_style = (np.prod(inp_sizes) * np.prod(tt_ranks)) ** (1.0 / inp_sizes.shape[0])
    return 0.1 / glarot_style * np.array(cores).astype(np.float32)

def matrix_to_tt(W, inp_sizes, out_sizes, tt_ranks):
    if False:
        while True:
            i = 10
    '\n    Convert a matrix into the TT-format.\n\n    This method will consume a 2D weight matrix such as those used in fully\n    connected layers in a neural network and will compute the TT-decomposition\n    of the weight matrix and return the TT-cores of the resulting computation.\n    This method should be used when converting a trained, fully connected layer,\n    into a TT-layer for increased speed and decreased parameter size. The size\n    of the ith core is:\n\n        tt_ranks[i] * inp_sizes[i] * out_sizes[i] * tt_ranks[i + 1].\n\n    Note that the following relationships of lengths of each input is expected:\n\n        len(inp_sizes) == len(out_sizes) == len(tt_ranks) - 1.\n\n    We also require that np.prod(inp_sizes) == W.shape[0] and that\n    np.prod(out_sizes) == W.shape[1].\n\n    Args:\n        W: two-dimensional weight matrix numpy array representing a fully\n           connected layer to be converted to TT-format; note that the weight\n           matrix is transposed before decomposed because we want to emulate the\n           X * W^T operation that the FC layer performs.\n        inp_sizes: list of the input dimensions of the respective cores\n        out_sizes: list of the output dimensions of the respective cores\n        tt_ranks: list of the ranks of the respective cores\n\n    Returns:\n        new_cores: One-dimensional list of cores concatentated along an axis\n   '
    assert len(inp_sizes) == len(out_sizes), 'The number of input dimensions (' + str(len(inp_sizes)) + ') must be equal to the number of output dimensions (' + str(len(out_sizes)) + ').'
    assert len(tt_ranks) == len(inp_sizes) + 1, 'The number of tt-ranks (' + str(len(tt_ranks)) + ') must be ' + 'one more than the number of input and output dimensions (' + str(len(out_sizes)) + ').'
    assert W.shape[0] == np.prod(inp_sizes), 'The product of the input sizes (' + str(np.prod(inp_sizes)) + ') must be equal to first dimension of W (' + str(W.shape[0]) + ').'
    assert W.shape[1] == np.prod(out_sizes), 'The product of the output sizes (' + str(np.prod(out_sizes)) + ') must be equal to second dimension of W (' + str(W.shape[1]) + ').'
    W = W.transpose()
    inp_sizes = np.array(inp_sizes)
    out_sizes = np.array(out_sizes)
    tt_ranks = np.array(tt_ranks)
    W_copy = W.copy()
    total_inp_size = inp_sizes.size
    W_copy = np.reshape(W_copy, np.concatenate((inp_sizes, out_sizes)))
    order = np.repeat(np.arange(0, total_inp_size), 2) + np.tile([0, total_inp_size], total_inp_size)
    W_copy = np.transpose(W_copy, axes=order)
    W_copy = np.reshape(W_copy, inp_sizes * out_sizes)
    cores = tt_svd(W_copy, inp_sizes * out_sizes, tt_ranks)
    new_cores = np.zeros(cores.shape).astype(np.float32)
    idx = 0
    for i in range(len(inp_sizes)):
        shape = (tt_ranks[i], inp_sizes[i], out_sizes[i], tt_ranks[i + 1])
        current_core = cores[idx:idx + np.prod(shape)].reshape(shape)
        current_core = current_core.transpose((1, 3, 0, 2))
        new_cores[new_cores.shape[0] - idx - np.prod(shape):new_cores.shape[0] - idx] = current_core.flatten()
        idx += np.prod(shape)
    return new_cores

def tt_svd(W, sizes, tt_ranks):
    if False:
        i = 10
        return i + 15
    '\n    Helper method for the matrix_to_tt() method performing the TT-SVD\n    decomposition.\n\n    Uses the TT-decomposition algorithm to convert a matrix to TT-format using\n    multiple reduced SVD operations.\n\n    Args:\n        W: two-dimensional weight matrix representing a fully connected layer to\n           be converted to TT-format preprocessed by the matrix_to_tt() method.\n        sizes: list of the dimensions of each of the cores\n        tt_ranks: list of the ranks of the respective cores\n\n    Returns:\n        cores: One-dimensional list of cores concatentated along an axis\n   '
    assert len(tt_ranks) == len(sizes) + 1
    C = W.copy()
    total_size = sizes.size
    core = np.zeros(np.sum(tt_ranks[:-1] * sizes * tt_ranks[1:]), dtype='float32')
    pos = 0
    for i in range(0, total_size - 1):
        shape = tt_ranks[i] * sizes[i]
        C = np.reshape(C, [shape, -1])
        (U, S, V) = np.linalg.svd(C, full_matrices=False)
        U = U[:, 0:tt_ranks[i + 1]]
        S = S[0:tt_ranks[i + 1]]
        V = V[0:tt_ranks[i + 1], :]
        core[pos:pos + tt_ranks[i] * sizes[i] * tt_ranks[i + 1]] = U.ravel()
        pos += tt_ranks[i] * sizes[i] * tt_ranks[i + 1]
        C = np.dot(np.diag(S), V)
    core[pos:pos + tt_ranks[total_size - 1] * sizes[total_size - 1] * tt_ranks[total_size]] = C.ravel()
    return core

def fc_net_to_tt_net(net):
    if False:
        for i in range(10):
            print('nop')
    pass