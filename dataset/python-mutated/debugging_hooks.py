from typing import Any
import torch
from torch.distributed import GradBucket
__all__ = ['noop_hook']

def noop_hook(_: Any, bucket: GradBucket) -> torch.futures.Future[torch.Tensor]:
    if False:
        while True:
            i = 10
    '\n    This DDP communication hook returns a future that wraps the input,\n    so it is a noop that does not incur any communication overheads.\n\n    This hook should **only** be used for headroom analysis of allreduce optimization,\n    instead of the normal gradient synchronization.\n    For example, if only less than 10% speedup of training time can be observed after this hook is registered,\n    it usually implies that allreduce is not a performance bottleneck for this case.\n    Such instrumentation can be particularly useful\n    if GPU traces cannot be easily retrieved or the trace analysis is complicated\n    some factors such as the overlap between allreduce and computation or the desynchronization across ranks.\n\n    Example::\n        >>> # xdoctest: +SKIP\n        >>> ddp_model.register_comm_hook(None, noop_hook)\n    '
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(bucket.buffer())
    return fut