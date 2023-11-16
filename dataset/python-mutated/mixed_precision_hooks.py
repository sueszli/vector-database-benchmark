import torch
import torch.distributed as dist
from torch.autograd import Variable
from dataclasses import dataclass
from typing import Any, no_type_check
from torch.distributed.utils import _free_storage

@dataclass
class _AllreduceUpcastHookState:
    """
    State to manage DDP mixed precision in backward / gradient communication.
    This contains a weakref to the DDP module for access to reducer and process
    group, and a stream to run parameter and gradient upcasts.
    """
    ddp_weakref: Any
    upcast_stream: torch.cuda.Stream
    wait_for_stream_enqueued: bool = False

@no_type_check
def _reducer_allreduce_and_upcast_hook(hook_state: _AllreduceUpcastHookState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    if False:
        while True:
            i = 10
    "\n    Performs allreduce in the reduced precision given by DDP's mixed precision\n    reduce_dtype, and upcasts parameters and gradients to fp32 in preparation\n    to run the optimizer.\n    "
    ddp_weakref = hook_state.ddp_weakref
    (reducer, process_group) = (ddp_weakref().reducer, ddp_weakref().process_group)
    gradient_is_bucket_view = ddp_weakref().gradient_as_bucket_view
    if ddp_weakref().mixed_precision.param_dtype != ddp_weakref().mixed_precision.reduce_dtype:
        bucket.set_buffer(bucket.buffer().to(ddp_weakref().mixed_precision.reduce_dtype))
    fut = reducer._run_allreduce_hook(bucket)
    ret_fut = torch.futures.Future()
    stream = hook_state.upcast_stream
    with torch.cuda.stream(stream):
        fut.wait()
        bucket.buffer().div_(process_group.size())
        ret_fut.set_result(bucket.buffer())
        (params, grads) = (bucket.parameters(), bucket.gradients())
        for (p, g) in zip(params, grads):
            p.data = p._fp_param
            _free_storage(p._mp_param)
            p.grad.data = p.grad.to(p.data.dtype)

    def wait_for_stream_cb():
        if False:
            while True:
                i = 10
        torch.cuda.current_stream().wait_stream(stream)
        for (n, p) in ddp_weakref().module.named_parameters():
            if hasattr(p, '_ddp_mp_hook_state'):
                p._ddp_mp_hook_state[1].remove()
                delattr(p, '_ddp_mp_hook_state')
            if not p.requires_grad and (not hasattr(p, '_ddp_ignored')):
                p.data = p._fp_param
        hook_state.wait_for_stream_enqueued = False
    if not hook_state.wait_for_stream_enqueued:
        Variable._execution_engine.queue_callback(wait_for_stream_cb)
        hook_state.wait_for_stream_enqueued = True
    return ret_fut