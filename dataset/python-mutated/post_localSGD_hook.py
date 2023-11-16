import logging
import torch
import torch.distributed as dist
from . import default_hooks as default
logger = logging.getLogger(__name__)

class PostLocalSGDState:
    """
    Stores the state for all-reducing gradients globally using ``process_group`` until step ``start_localSGD_iter``,
    and all-reducing gradients locally using ``subgroup`` afterwards.

    If ``process_group`` is ``None``, the global process group will be used.
    If ``subgroup`` is ``None``, the intra-node process group on each machine will be used.

    Additionally, ``post_local_gradient_allreduce`` may be worth tuning,
    because both true and false may give a faster convergence.
    """
    __slots__ = ['process_group', 'subgroup', 'start_localSGD_iter', 'post_local_gradient_allreduce', 'iter']

    def __init__(self, process_group, subgroup, start_localSGD_iter, post_local_gradient_allreduce=True):
        if False:
            return 10
        logger.info('Local SGD will be started after %s iterations', start_localSGD_iter)
        self.process_group = process_group
        self.subgroup = subgroup
        self.start_localSGD_iter = start_localSGD_iter
        self.post_local_gradient_allreduce = post_local_gradient_allreduce
        self.iter = 0

    def maybe_increase_iter(self, bucket):
        if False:
            print('Hello World!')
        if bucket.is_last():
            self.iter += 1
        if self.iter == self.start_localSGD_iter:
            logger.info('Start to apply local SGD after %s iterations.', self.iter)

def post_localSGD_hook(state: PostLocalSGDState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    if False:
        print('Hello World!')
    '\n    This DDP communication hook is used for running post-localSGD algorithm,\n    by combining with a model averaging component (e.g.,\n    :class:`~torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager`)\n    that runs after the optimizer step.\n\n    Args:\n        state (PostLocalSGDState): State information to run post-localSGD.\n            Users mainly need to tune ``start_localSGD_iter`` to determine when to start local SGD.\n        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.\n            Note that since DDP comm hook only supports single process single device mode,\n            only exactly one tensor is stored in this bucket.\n\n    Returns:\n        Future handler of the communication, which updates the gradients in place.\n\n    Example::\n        >>> # xdoctest: +SKIP\n        >>> state = PostLocalSGDState(process_group=process_group, subgroup=subgroup,\n                                  start_localSGD_iter=10)\n        >>> ddp_model.register_comm_hook(state, post_localSGD_hook)\n        >>> # Also need to establish a model averaging module and run model averaging after ``optimizer.step()``.\n        >>> # Please refer to the examples in ``torch.distributed.algorithms.model_averaging.averagers`` module.\n    '
    global_group_to_use = state.process_group if state.process_group is not None else dist.group.WORLD
    input_tensor = bucket.buffer()
    if state.iter < state.start_localSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(global_group_to_use, input_tensor)
    if not state.post_local_gradient_allreduce:
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut
    if state.subgroup is None:
        (state.subgroup, _) = dist.new_subgroups()
    return default._allreduce_fut(state.subgroup, input_tensor)