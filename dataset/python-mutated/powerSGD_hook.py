from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
__all__ = ['PowerSGDState', 'powerSGD_hook', 'batched_powerSGD_hook']
logger = logging.getLogger(__name__)

def _orthogonalize(matrices, epsilon=0):
    if False:
        for i in range(10):
            print('nop')
    "\n    Decide between Gram-Schmidt or QR factorization to orthogonalize a batch of matrices.\n    QR factorization doesn't work with half-precision, but it is usually faster with a rank > 2.\n    "
    assert len(matrices.shape) == 3 and matrices.shape[2] <= matrices.shape[1]
    num_matrices = matrices.shape[0]
    rank = matrices.shape[2]
    dtype = matrices.dtype
    if rank <= 2 or dtype in [torch.float16, torch.bfloat16]:
        _orthogonalize_gram_schmidt(matrices, epsilon=epsilon)
    else:
        torch.linalg.qr(matrices, out=(matrices, torch.empty(num_matrices, rank, rank, device=matrices.device, dtype=dtype)))

def _orthogonalize_gram_schmidt(matrices, epsilon=0):
    if False:
        i = 10
        return i + 15
    '\n    Applies Gram-Schmidt procedure to orthogonalize a batch of matrices.\n    If epsilon is 0, this is equivalent to `torch.qr(matrices, out=(matrices, _))`,\n    '
    num_cols = matrices.shape[2]
    for i in range(num_cols):
        col = matrices[:, :, i:i + 1]
        if epsilon == 0:
            try:
                col /= torch.norm(col, dim=1, keepdim=True)
            except ZeroDivisionError:
                logger.error('The matrices to be orthogonalized has at least a column of all 0s. Please set a small value such as 1e-8 as `orthogonalization_epsilon` in PowerSGD state.')
                col.fill_(0.0)
        else:
            col /= torch.norm(col, dim=1, keepdim=True) + epsilon
        if i + 1 < num_cols:
            rest = matrices[:, :, i + 1:]
            rest -= torch.sum(col * rest, dim=1, keepdim=True) * col

def _should_compress(num_rows, num_cols, matrix_approximation_rank, min_compression_rate):
    if False:
        return 10
    '\n    Returns a recommendation as to whether the 2D tensor described by the arguments is worth compressing,\n    including statistics describing the expected savings from compression.  We consider a tensor worth\n    compressing when ``min_compression_rate`` < uncompressed size / compressed size, where\n    uncompressed size = ``num_rows`` * ``num_cols``,\n    and compressed size = (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.\n\n    The result of this function is a tuple of the form (compression_recommendation, uncompressed_el_count, compressed_el_count), where:\n\n    compression_recommendation is true if the tensor is worth compressing, and false otherwise (see above);\n\n    uncompressed_el_count is the uncompressed element count, i.e. ``num_rows`` * ``num_cols``; and,\n\n    compress_el_count is the element count after compression, i.e. (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.\n    '
    uncompressed_size = num_rows * num_cols
    compressed_size = (num_rows + num_cols) * matrix_approximation_rank
    return (compressed_size * min_compression_rate < uncompressed_size, uncompressed_size, compressed_size)

def _report_compression_stats(bucket, state):
    if False:
        for i in range(10):
            print('nop')
    '\n    Report compression stats at the frequency of `compression_stats_logging_frequency` specified in PowerSGD state.\n    '
    if bucket.is_last() and state.iter >= state.next_stats_report:
        stats = state.compression_stats()
        logger.info('Compression stats: iter %s, total before compression %s, total after compression %s, rate %s', state.iter, stats[1], stats[2], stats[0])
        state.next_stats_report = state.iter + state.compression_stats_logging_frequency

class PowerSGDState:
    """
    Stores both the algorithm's hyperparameters and the internal state for all the gradients during the training.
    Particularly, ``matrix_approximation_rank`` and ``start_powerSGD_iter`` are the main hyperparameters that should be tuned by the user.
    For performance, we suggest to keep binary hyperparameters ``use_error_feedback`` and ``warm_start`` on.

    1. ``matrix_approximation_rank`` controls the size of compressed low-rank tensors, which determines the compression rate. The lower the rank, the stronger the compression.

        1.1. If ``matrix_approximation_rank`` is too low, the full model quality will need more training steps to reach or will never reach and yield loss in accuracy.

        1.2. The increase of ``matrix_approximation_rank`` can substantially increase the computation costs of the compression, and the accuracy may not be further improved beyond a certain ``matrix_approximation_rank`` threshold.

    To tune ``matrix_approximation_rank``, we suggest to start from 1 and increase by factors of 2 (like an exponential grid search, 1, 2, 4, ...), until a satisfactory accuracy is reached. Typically only a small value 1-4 is used. For some NLP tasks (as shown in Appendix D of the original paper), this value has been increased to 32.

    2. ``start_powerSGD_iter`` defers PowerSGD compression until step ``start_powerSGD_iter``, and vanilla allreduce runs prior to step ``start_powerSGD_iter``. This hybrid scheme of **vanilla allreduce + PowerSGD** can effectively improve the accuracy, even a relatively small ``matrix_approximation_rank`` is used. This is because that, the beginning of training phase is usually very sensitive to inaccurate gradients, and compressing gradients too early may make the training quickly take a suboptimal trajectory, which can result in an irrecoverable impact on the accuracy.

    To tune ``start_powerSGD_iter``, we suggest to start with 10% of total training steps, and increase it until a satisfactory accuracy is reached. If there is a warm-up stage in the training, ``start_powerSGD_iter`` typically should be no less than the number of warm-up steps.

    3. ``min_compression_rate`` is the minimum compression rate required when a layer is compressed. Due to the computation overheads incurred by the compression, a tensor is worth compressing only if there can be sufficient saving in bandwidth, where ``(num_rows + num_cols) * matrix_approximation_rank * min_compression_rate < num_rows * num_cols``. If the specified compression rate threshold cannot be satisfied, the tensor will be directly allreduced without compression.

    Compression statistics are logged every ``compression_stats_logging_frequency`` iterations once PowerSGD compression starts.

    4. ``orthogonalization_epsilon`` can be a very small value (e.g., 1e-8) added to every normalized matrix column in orthogonalization step, to prevent div-by-zero error if any column has all 0s. If this can already be prevented (e.g., by batch normalization), an epsilon of 0 is recommended for accuracy.

    5. ``batch_tensors_with_same_shape`` controls whether to compress and decompress tensors with same shape in a batched operation to achieve higher parallelism. Note that you should also increase the bucket size (i.e., ``bucket_cap_mb`` arg in DDP constructor) to make more same-shaped tensors appear in the same bucket, however this may reduce the overlap between computation and communication, and increase the memory footprint due to stacking the tensors of the same shape. Set to ``True`` if the compression / decompression computation is a bottleneck.

    .. warning ::
        If error feedback or warm-up is enabled, the minimum value of ``start_powerSGD_iter`` allowed in DDP is 2.
        This is because there is another internal optimization that rebuilds buckets at iteration 1 in DDP,
        and this can conflict with any tensor memorized before the rebuild process.
    """
    __slots__ = ['process_group', 'matrix_approximation_rank', 'start_powerSGD_iter', 'min_compression_rate', 'orthogonalization_epsilon', 'use_error_feedback', 'warm_start', 'batch_tensors_with_same_shape', 'rng', 'error_dict', 'p_memory_dict', 'q_memory_dict', 'iter', 'total_numel_before_compression', 'total_numel_after_compression', 'compression_stats_logging_frequency', 'next_stats_report']

    def __init__(self, process_group, matrix_approximation_rank=1, start_powerSGD_iter=1000, min_compression_rate=2, use_error_feedback=True, warm_start=True, orthogonalization_epsilon=0, random_seed=0, compression_stats_logging_frequency=10000, batch_tensors_with_same_shape: bool=False):
        if False:
            print('Hello World!')
        logger.info('PowerSGD config: matrix_approximation_rank = %s; start_powerSGD_iter = %s; min_compression_rate = %s; orthogonalization_epsilon = %s; use_error_feedback = %s; warm_start = %s; random_seed = %s; compression_stats_logging_frequency = %s; batch_tensors_with_same_shape = %s', matrix_approximation_rank, start_powerSGD_iter, min_compression_rate, orthogonalization_epsilon, use_error_feedback, warm_start, random_seed, compression_stats_logging_frequency, batch_tensors_with_same_shape)
        self.process_group = process_group
        self.matrix_approximation_rank = matrix_approximation_rank
        if (use_error_feedback or warm_start) and start_powerSGD_iter <= 1:
            raise ValueError('Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, because PowerSGD can only be applied after the first two iterations in DDP.')
        self.start_powerSGD_iter = start_powerSGD_iter
        self.min_compression_rate = min_compression_rate
        self.use_error_feedback = use_error_feedback
        self.warm_start = warm_start
        self.orthogonalization_epsilon = orthogonalization_epsilon
        import numpy as np
        self.rng = np.random.RandomState(random_seed)
        self.error_dict: Dict[int, torch.Tensor] = {}
        self.p_memory_dict: Dict[int, torch.Tensor] = {}
        self.q_memory_dict: Dict[int, torch.Tensor] = {}
        self.iter = 0
        self.total_numel_before_compression = 0
        self.total_numel_after_compression = 0
        self.compression_stats_logging_frequency = max(1, compression_stats_logging_frequency)
        self.next_stats_report = 0
        self.batch_tensors_with_same_shape = batch_tensors_with_same_shape

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a ``Dict[str, Any]`` which will be pickled and saved.\n        ``process_group`` is not serializable and excluded from\n        a returned state.\n        '
        logger.warning('NOTE: Process group is not serializable and excluded from a saved state.')
        return {slot: getattr(self, slot) for slot in self.__slots__ if slot != 'process_group'}

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        '\n        Takes a provided ``state`` and retrieves ``PowerSGDState``.\n        ``process_group`` is set to default.\n        '
        self.process_group = distributed_c10d._get_default_group()
        logger.warning('NOTE: Process group will be set to a default group (i.e. the world size).                If a different group is desired, please set `self.process_group` after PowerSGD state is loaded.')
        for (slot, value) in state.items():
            setattr(self, slot, value)

    def maybe_increase_iter(self, bucket):
        if False:
            while True:
                i = 10
        if bucket.is_last():
            self.iter += 1
        if self.iter == self.start_powerSGD_iter:
            logger.info('Start to apply PowerSGD after %s iterations.', self.iter)

    def compression_stats(self):
        if False:
            return 10
        '\n        Returns the latest compression statistics as a tuple of the form (compress_rate, numel_before_compression, numel_after_compression), where:\n\n        compress_rate is the effective compression rate i.e. (number of elements before compression) / (number of elements after compression);\n\n        numel_before_compression is the total number of elements before compression was applied; and,\n\n        numel_after_compression is the total number of elements after compression was applied.\n        '
        compress_rate = self.total_numel_before_compression / self.total_numel_after_compression if self.total_numel_after_compression > 0 else 0
        return (compress_rate, self.total_numel_before_compression, self.total_numel_after_compression)

def powerSGD_hook(state: PowerSGDState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    '\n    This DDP communication hook implements PowerSGD gradient compression\n    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.\n    Once gradient tensors are aggregated across all workers, this hook applies\n    compression as follows:\n\n    1. Views the input flattened 1D gradient tensor as a list of per-parameter tensors, and divides all the tensors into two groups:\n\n        1.1 The tensors that should be compressed before allreduce, because the compression can give enough saving in bandwidth.\n\n        1.2 Rest of the tensors will be directly allreduced without compression, including all the vector tensors (for biases).\n\n    2. Handles uncompressed tensors:\n\n        2.1. Allocate contiguous memory for those uncompressed tensors, and allreduces all the uncompressed tensors as a batch, without compression;\n\n        2.2. Copies the individual uncompressed tensors from the contiguous memory back to the input tensor.\n\n    3. Handles the tensors that should be compressed by PowerSGD compression:\n\n        3.1. For each tensor M, creates two low-rank tensors P and Q for decomposing M,\n        such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;\n\n        3.2. Computes each P in Ps, which is equal to MQ;\n\n        3.3. Allreduces Ps as a batch;\n\n        3.4. Orthogonalizes each P in Ps;\n\n        3.5. Computes each Q in Qs, which is approximately equal to M^TP;\n\n        3.6. Allreduces Qs as a batch;\n\n        3.7. Computes each M among all the compressed tensors, which is approximately equal to PQ^T.\n\n    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.\n    This not only gives the user more control over the tradeoff between speedup and accuracy,\n    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.\n\n    Args:\n        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.\n            To tune the compression configs, mainly need to tune ``matrix_approximation_rank``, ``start_powerSGD_iter``\n            and ``min_compression_rate``.\n        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.\n            Note that since DDP comm hook only supports single process single device mode,\n            only exactly one tensor is stored in this bucket.\n\n    Returns:\n        Future handler of the communication, which updates the gradients in place.\n\n    Example::\n        >>> # xdoctest: +SKIP\n        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1,\n                                  start_powerSGD_iter=10, min_compression_rate=0.5)\n        >>> ddp_model.register_comm_hook(state, powerSGD_hook)\n    '
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    input_tensor = bucket.buffer()
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)
    device = input_tensor.device
    dtype = input_tensor.dtype
    bucket_index = bucket.index()
    input_tensor_cp = None
    total_length = input_tensor.shape[0]
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logger.info('A zero tensor of length %s that represents local error is created.', total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)
        input_tensor_cp = torch.clone(input_tensor).detach()
    tensors = bucket.gradients()
    (tensors_to_compress, uncompressed_tensors) = ([], [])
    total_Ps_size = 0
    total_Qs_size = 0
    for tensor in tensors:
        matrix = tensor.view(tensor.shape[0], -1)
        (n, m) = matrix.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        compress_test = _should_compress(n, m, matrix_approximation_rank, state.min_compression_rate)
        state.total_numel_before_compression += compress_test[1]
        if compress_test[0]:
            tensors_to_compress.append(matrix)
            total_Ps_size += n * matrix_approximation_rank
            total_Qs_size += m * matrix_approximation_rank
            state.total_numel_after_compression += compress_test[2]
        else:
            uncompressed_tensors.append(tensor)
            state.total_numel_after_compression += compress_test[1]
    _report_compression_stats(bucket, state)
    uncompressed_tensors_memory = torch.cat([tensor.view(-1) for tensor in uncompressed_tensors]) if uncompressed_tensors else torch.tensor([], device=device, dtype=dtype)
    need_randomize_qs = False
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        need_randomize_qs = True
        if state.warm_start:
            logger.info('Allocating contiguous memory of length %s for Ps, and of length %s for Qs, respectively.', total_Ps_size, total_Qs_size)
        state.p_memory_dict[bucket_index] = torch.empty(total_Ps_size, device=device, dtype=dtype)
        state.q_memory_dict[bucket_index] = torch.empty(total_Qs_size, device=device, dtype=dtype)
    shape_to_tensors = defaultdict(list)
    for tensor in tensors_to_compress:
        shape_to_tensors[tensor.shape].append(tensor)

    def maybe_batched_tensors_to_compress():
        if False:
            while True:
                i = 10
        for tensors in shape_to_tensors.values():
            if state.batch_tensors_with_same_shape:
                batch_size = len(tensors)
                if batch_size == 1:
                    yield tensors[0].unsqueeze(0)
                else:
                    yield torch.stack(tensors)
            else:
                for tensor in tensors:
                    yield tensor.unsqueeze(0)
    tensors_to_compress = []
    ps = []
    qs = []
    p_idx = 0
    q_idx = 0
    for tensor in maybe_batched_tensors_to_compress():
        (batch_size, n, m) = tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        tensors_to_compress.append(tensor)
        ps.append(state.p_memory_dict[bucket_index][p_idx:p_idx + batch_size * n * matrix_approximation_rank].view(batch_size, n, matrix_approximation_rank))
        qs.append(state.q_memory_dict[bucket_index][q_idx:q_idx + batch_size * m * matrix_approximation_rank].view(batch_size, m, matrix_approximation_rank))
        p_idx += batch_size * n * matrix_approximation_rank
        q_idx += batch_size * m * matrix_approximation_rank
    if not need_randomize_qs:
        for q in qs:
            _orthogonalize(q, state.orthogonalization_epsilon)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(state.rng.randint(1000000000))
            for q in qs:
                q.copy_(torch.randn(*q.shape, device='cpu', dtype=dtype))
                _orthogonalize(q, state.orthogonalization_epsilon)
    for (tensor, q, p) in zip(tensors_to_compress, qs, ps):
        torch.bmm(tensor, q, out=p)
    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(uncompressed_tensors_memory, group=group_to_use, async_op=True).get_future()

    def unpack_uncompressed_tensors_and_allreduce_ps(fut):
        if False:
            print('Hello World!')
        uncompressed_tensors_memory = fut.value()[0].div_(world_size)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(uncompressed_tensors_memory[idx:idx + tensor.numel()].view_as(tensor))
            idx += tensor.numel()
        return dist.all_reduce(state.p_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future().wait()[0]

    def compute_qs(fut):
        if False:
            for i in range(10):
                print('nop')
        state.p_memory_dict[bucket_index] = fut.value()
        for p in ps:
            _orthogonalize(p, state.orthogonalization_epsilon)
        for (tensor, p, q) in zip(tensors_to_compress, ps, qs):
            torch.bmm(tensor.transpose(1, 2), p, out=q)
        return dist.all_reduce(state.q_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future().wait()[0]

    def decompress(fut):
        if False:
            print('Hello World!')
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)
        for (p, q, tensor) in zip(ps, qs, tensors_to_compress):
            torch.bmm(p, q.transpose(1, 2), out=tensor)
        if state.batch_tensors_with_same_shape:
            for tensor in tensors_to_compress:
                if tensor.shape[0] == 1:
                    continue
                original_tensors = shape_to_tensors[tensor.shape[1:]]
                for (i, original_tensor) in enumerate(original_tensors):
                    original_tensor.copy_(tensor[i])
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        if state.use_error_feedback:
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()
        state.maybe_increase_iter(bucket)
        return input_tensor
    return allreduce_contiguous_uncompressed_tensors_fut.then(unpack_uncompressed_tensors_and_allreduce_ps).then(compute_qs).then(decompress)

def batched_powerSGD_hook(state: PowerSGDState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    if False:
        return 10
    '\n    This DDP communication hook implements a simplified PowerSGD gradient compression\n    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.\n    This variant does not compress the gradients layer by layer,\n    but instead compresses the flattened input tensor that batches all the gradients.\n    Therefore, it is **faster** than :meth:`powerSGD_hook`,\n    but usually results in a **much lower accuracy**, unless ``matrix_approximation_rank`` is 1.\n\n    .. warning ::\n        Increasing ``matrix_approximation_rank`` here may not necessarily increase the accuracy,\n        because batching per-parameter tensors without column/row alignment can destroy low-rank structure.\n        Therefore, the user should always consider :meth:`powerSGD_hook` first,\n        and only consider this variant when a satisfactory accuracy can be achieved when ``matrix_approximation_rank`` is 1.\n\n    Once gradient tensors are aggregated across all workers, this hook applies\n    compression as follows:\n\n    1. Views the input flattened 1D gradient tensor as a square-shaped tensor M with 0 paddings;\n\n    2. Creates two low-rank tensors P and Q for decomposing M, such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;\n\n    3. Computes P, which is equal to MQ;\n\n    4. Allreduces P;\n\n    5. Orthogonalizes P;\n\n    6. Computes Q, which is approximately equal to M^TP;\n\n    7. Allreduces Q;\n\n    8. Computes M, which is approximately equal to PQ^T.\n\n    9. Truncates the input tensor to the original length.\n\n    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.\n    This not only gives the user more control over the tradeoff between speedup and accuracy,\n    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.\n\n    Args:\n        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.\n            To tune the compression configs, mainly need to tune ``matrix_approximation_rank`` and ``start_powerSGD_iter``.\n        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.\n            Note that since DDP comm hook only supports single process single device mode,\n            only exactly one tensor is stored in this bucket.\n\n    Returns:\n        Future handler of the communication, which updates the gradients in place.\n\n    Example::\n        >>> # xdoctest: +SKIP\n        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1)\n        >>> ddp_model.register_comm_hook(state, batched_powerSGD_hook)\n    '
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    input_tensor = bucket.buffer()
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)
    device = input_tensor.device
    total_length = input_tensor.shape[0]
    state.total_numel_before_compression += total_length
    square_side_length = math.ceil(math.sqrt(total_length))
    state.total_numel_after_compression += square_side_length * state.matrix_approximation_rank * 2
    padded_total_length = square_side_length ** 2
    input_tensor.resize_(padded_total_length)
    input_tensor[total_length:padded_total_length].fill_(0)
    _report_compression_stats(bucket, state)
    bucket_index = bucket.index()
    input_tensor_cp = None
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logger.info('A zero tensor of length %s that represents local error is created.', padded_total_length)
            state.error_dict[bucket_index] = torch.zeros(padded_total_length, device=device, dtype=input_tensor.dtype)
        input_tensor_cp = torch.clone(input_tensor).detach()
    matrix = input_tensor.view(square_side_length, square_side_length)
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        if state.warm_start:
            logger.info('Initializing low-rank tensors P and Q, each of which has a shape of %s x %s.', square_side_length, state.matrix_approximation_rank)

        def create_low_rank_tensor(fill_random_values, rng):
            if False:
                while True:
                    i = 10
            'Returns a low-rank 2D tensor of square_side_length * matrix_approximation_rank.'
            if fill_random_values:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(rng.randint(1000000000))
                    return torch.randn(square_side_length, state.matrix_approximation_rank, device='cpu', dtype=input_tensor.dtype).to(device)
            else:
                return torch.empty(square_side_length, state.matrix_approximation_rank, device=device, dtype=input_tensor.dtype)
        state.p_memory_dict[bucket_index] = create_low_rank_tensor(fill_random_values=False, rng=state.rng)
        state.q_memory_dict[bucket_index] = create_low_rank_tensor(fill_random_values=True, rng=state.rng)
    _orthogonalize(state.q_memory_dict[bucket_index])
    torch.matmul(matrix, state.q_memory_dict[bucket_index], out=state.p_memory_dict[bucket_index])
    allreduce_p_fut = dist.all_reduce(state.p_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future()

    def compute_q(fut):
        if False:
            for i in range(10):
                print('nop')
        state.p_memory_dict[bucket_index] = fut.value()[0]
        _orthogonalize(state.p_memory_dict[bucket_index])
        torch.matmul(matrix.t(), state.p_memory_dict[bucket_index], out=state.q_memory_dict[bucket_index])
        return dist.all_reduce(state.q_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future().wait()[0]

    def decompress(fut):
        if False:
            print('Hello World!')
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)
        torch.matmul(state.p_memory_dict[bucket_index], state.q_memory_dict[bucket_index].t(), out=matrix)
        if state.use_error_feedback:
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()
        ret = input_tensor.resize_(total_length)
        state.maybe_increase_iter(bucket)
        return ret
    return allreduce_p_fut.then(compute_q).then(decompress)