import logging
import os
import signal
import threading
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fairseq.distributed import DistributedTimeoutWrapper, LegacyDistributedDataParallel, ModuleProxyWrapper, TPUDistributedDataParallel
logger = logging.getLogger(__name__)
_SLOWMO_DDP_DISABLED = False
try:
    from fairscale.experimental.nn.data_parallel import SlowMoBaseAlgorithm, SlowMoDistributedDataParallel
except ImportError:
    _SLOWMO_DDP_DISABLED = True

def DistributedFairseqModel(args, model, process_group, device):
    if False:
        print('Hello World!')
    '\n    Wrap a *model* to support distributed data parallel training.\n\n    This is similar to the built-in DistributedDataParallel, but allows\n    additional configuration of the DistributedDataParallel class to\n    use, and also provides easier access to the wrapped model by\n    forwarding requests for missing attributes to the wrapped model.\n\n    Args:\n        args (argparse.Namespace): fairseq args\n        model (BaseFairseqModel): model to wrap\n        process_group: the c10d process group to be used for distributed data\n            parallel all-reduction.\n        device: device to move model to\n    '
    assert isinstance(model, nn.Module)
    if args.tpu:
        wrapped_model = TPUDistributedDataParallel(module=model.to(device), process_group=process_group)
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend in {'c10d', 'pytorch_ddp'}:
        wrapped_model = DistributedDataParallel(module=model.to(device), device_ids=[args.device_id], output_device=args.device_id, broadcast_buffers=args.broadcast_buffers, bucket_cap_mb=args.bucket_cap_mb, process_group=process_group, find_unused_parameters=args.find_unused_parameters, gradient_as_bucket_view=args.gradient_as_bucket_view)
        if args.ddp_comm_hook == 'fp16':
            logger.info('enable fp16 communication hook in DDP')
            try:
                from torch.distributed.algorithms.ddp_comm_hooks import DDPCommHookType, register_ddp_comm_hook
            except:
                logger.error('Could not import from torch.distributed.algorithms.ddp_comm_hooks; you may need to update your pytorch version')
                raise
            register_ddp_comm_hook(DDPCommHookType.FP16_COMPRESS, wrapped_model)
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend in {'no_c10d', 'legacy_ddp'}:
        wrapped_model = LegacyDistributedDataParallel(module=model.to(device), buffer_size=2 ** 28, process_group=process_group)
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend == 'slowmo':
        if _SLOWMO_DDP_DISABLED:
            raise ImportError('Cannot find SlowMoDistributedDataParallel. Please install fairscale with: pip install fairscale')
        if args.slowmo_momentum is None:
            if args.distributed_world_size <= 16:
                args.slowmo_momentum = 0.0
            elif args.distributed_world_size <= 32:
                args.slowmo_momentum = 0.2
            elif args.distributed_world_size <= 64:
                args.slowmo_momentum = 0.5
            else:
                args.slowmo_momentum = 0.6
        slowmo_base_algorithm = SlowMoBaseAlgorithm[args.slowmo_base_algorithm.upper()]
        wrapped_model = SlowMoDistributedDataParallel(module=model.to(device), broadcast_buffers=args.broadcast_buffers, nprocs_per_node=args.nprocs_per_node, slowmo_momentum=args.slowmo_momentum, slowmo_base_algorithm=slowmo_base_algorithm, localsgd_frequency=args.localsgd_frequency)
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend == 'fully_sharded':
        try:
            from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
        except ImportError:
            raise ImportError('Cannot find FullyShardedDataParallel. Please install fairscale with: pip install fairscale')
        assert isinstance(model, FSDP), 'expected model to already be wrapped in FSDP'
        wrapped_model = model
        if args.memory_efficient_fp16:
            wrapped_model = wrapped_model.half()
        if not args.cpu_offload:
            wrapped_model = wrapped_model.to(device=device)
    else:
        raise ValueError('Unknown --ddp-backend: ' + args.ddp_backend)
    if getattr(args, 'heartbeat_timeout', -1) > 0:
        wrapped_model = DistributedTimeoutWrapper(wrapped_model, timeout=getattr(args, 'heartbeat_timeout', -1))
    return wrapped_model