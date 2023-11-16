import torch
import torch.distributed as dist
from torch import nn

def _quantize_per_tensor_cuda(x, scale, zero_point):
    if False:
        i = 10
        return i + 15
    y = torch.round(x / scale) + zero_point
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y

def _dequantize_per_tensor_cuda(y, scale, zero_point):
    if False:
        while True:
            i = 10
    x = scale * (y.to(torch.float32) - zero_point)
    return x

def _quantize_per_channel_cuda(x, scale, zero_point):
    if False:
        i = 10
        return i + 15
    y = torch.zeros(x.size(), device=x.device)
    for i in range(x.size()[0]):
        y[i, :] = torch.round(x[i, :] / scale[i]) + zero_point[i]
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y

def _dequantize_per_channel_cuda(y, scale, zero_point):
    if False:
        for i in range(10):
            print('nop')
    y = y.to(torch.float32).cuda(y.device)
    x = torch.zeros_like(y, device=y.device)
    for i in range(x.size()[0]):
        x[i, :] = scale[i] * (y[i, :] - zero_point[i])
    return x

def _get_allgather_out_list(all_gather_in_list, world_size):
    if False:
        i = 10
        return i + 15
    out_list = [torch.zeros_like(all_gather_in_list, device=all_gather_in_list.device, dtype=all_gather_in_list.dtype) for _ in range(world_size)]
    return out_list

def quantization_pertensor_hook(process_group: dist.ProcessGroup, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    if False:
        return 10
    "\n    Applies the ``torch.quantize_per_tensor`` logic to DDP using ``allgather``\n    protocol. Workers first allgather the scale and zero point of their own\n    ``GradBucket`` prior to the quantization. After all workers have that information,\n    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's\n    own gradient tensor, and uses ``allgather`` to communicate these across all workers.\n    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes and\n    aggregates each quantized gradient tensor locally and returns the mean.\n\n    .. warning ::\n        This is experimental, and uses ``allgather`` protocol which is considerably slower than\n        ``allreduce`` protocol. It works only with flattened grads.\n\n    Example::\n        >>> # xdoctest: +SKIP\n        >>> ddp_model.register_comm_hook(process_group, quantization_pertensor_hook)\n    "
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    world_size = group_to_use.size()
    tensor = bucket.buffer()
    myObserver = torch.ao.quantization.MinMaxObserver().cuda(tensor.device)
    myObserver(tensor)
    (s, z) = myObserver.calculate_qparams()
    s_and_z = torch.FloatTensor([s, z]).cuda(tensor.device)
    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)
    fut = dist.all_gather(all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True).get_future()

    def quantize_and_allgather(fut):
        if False:
            while True:
                i = 10
        all_ranks_s_and_z = fut.wait()[0]
        quantized_tensor = _quantize_per_tensor_cuda(tensor, all_ranks_s_and_z[rank][0], all_ranks_s_and_z[rank][1])
        fut = dist.all_gather(_get_allgather_out_list(quantized_tensor, world_size), quantized_tensor, group=group_to_use, async_op=True).get_future()
        return fut.wait()

    def dequantize_and_aggregate(fut):
        if False:
            print('Hello World!')
        all_ranks_quantized_tensor = fut.wait()[0]
        aggregated_dequantized_tensor = torch.zeros_like(all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32)
        for (r, quantized_tensor) in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_tensor_cuda(quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1])
        return aggregated_dequantized_tensor / world_size
    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)

def quantization_perchannel_hook(process_group: dist.ProcessGroup, bucket: dist.GradBucket, bucket_size=512) -> torch.futures.Future[torch.Tensor]:
    if False:
        return 10
    "\n    Applies the ``torch.quantize_per_channel`` logic to DDP using ``allgather``\n    protocol. Compared to pertensor, the main motivation of perchannel is\n    for considerably large tensors such as a tensor that contains 6 million\n    elements quantizing per a bucket size of 512 (or 128) elements may significantly\n    increase the resolution.\n\n    It first splits ``GradBucket`` tensor into multiple chunks (channels) of ``bucket_size``\n    elements. Then, workers allgather the scales and zero points of their own\n    ``GradBucket`` prior to the quantization. After all workers have that information,\n    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's\n    own gradient tensor, and uses ``allgather`` to communicate these across all workers.\n    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes, flattens, and\n    aggregates each quantized gradient tensor locally and returns the mean.\n\n    .. warning ::\n        This is experimental, and uses ``allgather`` protocol which is considerably slower than\n        ``allreduce`` protocol. It works only with flattened grads.\n\n    Example::\n        >>> # xdoctest: +SKIP\n        >>> ddp_model.register_comm_hook(process_group, quantization_perchannel_hook)\n    "
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    world_size = group_to_use.size()
    tensor = bucket.buffer()
    tensor_in_channels = nn.functional.pad(input=tensor, pad=(0, bucket_size - len(tensor) % bucket_size), mode='constant', value=0).view(-1, bucket_size).cuda(tensor.device)
    myPerChannelObserver = torch.ao.quantization.PerChannelMinMaxObserver().cuda(tensor.device)
    myPerChannelObserver(tensor_in_channels)
    (s_ch, z_ch) = myPerChannelObserver.calculate_qparams()
    s_and_z = torch.stack((s_ch, z_ch)).cuda(tensor.device)
    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)
    fut = dist.all_gather(all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True).get_future()

    def quantize_and_allgather(fut):
        if False:
            i = 10
            return i + 15
        all_ranks_s_and_z = fut.wait()[0]
        quantized_tensor = _quantize_per_channel_cuda(tensor_in_channels, all_ranks_s_and_z[rank, 0, :], all_ranks_s_and_z[rank, 1, :])
        fut = dist.all_gather(_get_allgather_out_list(quantized_tensor, world_size), quantized_tensor, group=group_to_use, async_op=True).get_future()
        return fut.wait()

    def dequantize_and_aggregate(fut):
        if False:
            for i in range(10):
                print('nop')
        all_ranks_quantized_tensor = fut.wait()[0]
        aggregated_dequantized_tensor = torch.zeros_like(all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32)
        for (r, quantized_tensor) in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_channel_cuda(quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1])
        return torch.flatten(aggregated_dequantized_tensor).cuda(tensor.device)[:tensor.size()[0]] / world_size
    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)