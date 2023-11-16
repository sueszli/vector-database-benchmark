import logging
import os
import paddle
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import GroupShardedOptimizerStage2
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import GroupShardedStage2
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage3 import GroupShardedStage3
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import GroupShardedScaler
from paddle.distributed.fleet.utils.mix_precision_utils import MixPrecisionOptimizer
from paddle.distributed.utils.log_utils import get_logger
from paddle.optimizer import Optimizer
logger_ = get_logger(logging.WARNING)

def group_sharded_parallel(model, optimizer, level, scaler=None, group=None, offload=False, sync_buffers=False, buffer_max_size=2 ** 23, segment_size=2 ** 20, sync_comm=False, dp_group=None, exclude_layer=None):
    if False:
        i = 10
        return i + 15
    '\n    Use group_sharded_parallel can perform group shared configuration on the model, optimizer and GradScaler. Level has three string options, \'os\', \'os_g\' and \'p_g_os\' corresponds to three different usage scenarios: optimizer state segmentation, optimizer state + gradient segmentation, and parameter + gradient + optimizer state segmentation.\n    Usually, optimizer state + gradient segmentation is actually a re optimization of optimizer state segmentation, so optimizer state + gradient segmentation can be used to realize optimizer state segmentation.\n\n    Args:\n        model (Layer): The layer to be wrapped with group_sharded_parallel.\n        optimizer (Optimizer): The optimizer to be wrapped with group_sharded_parallel.\n        level (str): The different level of the group sharded. Such as `os`, `os_g`, `p_g_os`.\n        scaler (GradScaler, optional): If AMP is used, you need to pass GradScaler. Defaults to None, indicating that GradScaler is not used.\n        group (Group, optional): The group instance. Defaults to None, indicating that the default environment group is used.\n        offload (bool, optional): Whether to use the offload function. Defaults to False, which means that the offload function is not used.\n        sync_buffers (bool, optional): Whether to broadcast model buffers. It is generally used when there are registered model buffers. Defaults to False, indicating that model buffers are not used.\n        buffer_max_size (int, optional): The max size of the buffer used to integrate gradient in `os_g`. The larger the size, the more GPU memory will be used. Defaults to 2**23, which means that the dimension of the buffer is 2**23.\n        segment_size (int, optional): The smallest size of parameter to be sharded in `p_g_os`. Defaults to 2**20, indicating that the dimension of the minimum segmented parameter is 2**20.\n        sync_comm (bool, optional): Whether to use synchronous communication, only in `p_g_os` used. Defaults to False, indicating that asynchronous communication is used.\n        dp_group(Group, optional): dp communication group, support to combine stage2 or stage3 with dp hybrid communication.\n        exclude_layer(list, optional): exclude some layers for slicing for sharding stage3, for example, exclude_layer=["GroupNorm", id(model.gpt.linear)], exclude_layer must contain the layers\' name or one layer\'s id.\n\n    Returns:\n        model: A wrapper for group sharded given model.\n        optimizer: A wrapper for group sharded given optimizer.\n        scaler: A wrapper for group sharded given scaler.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n            >>> import paddle\n            >>> from paddle.nn import Linear\n            >>> from paddle.distributed import fleet\n            >>> from paddle.distributed.sharding import group_sharded_parallel\n\n            >>> fleet.init(is_collective=True)\n            >>> group = paddle.distributed.new_group([0, 1])\n            >>> model = Linear(1000, 1000)\n\n            >>> clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)\n            >>> optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters(), weight_decay=0.00001, grad_clip=clip)\n\n            >>> # wrap sharding model, optimizer and scaler\n            >>> model, optimizer, scaler = group_sharded_parallel(model, optimizer, "p_g", scaler=scaler)\n\n            >>> img, label = data\n            >>> label.stop_gradient = True\n            >>> img.stop_gradient = True\n\n            >>> out = model(img)\n            >>> loss = paddle.nn.functional.cross_entropy(input=out, label=label)\n\n            >>> loss.backward()\n            >>> optimizer.step()\n            >>> optimizer.clear_grad()\n\n    '
    device = paddle.get_device().split(':')[0]
    assert device in ['gpu', 'xpu'], 'group_sharded_parallel only support gpu and xpu now'
    assert isinstance(model, paddle.nn.Layer), 'The model must be the instance of paddle.nn.Layer.'
    assert isinstance(optimizer, (MixPrecisionOptimizer, Optimizer)), 'The optimizer must be the instance of paddle.optimizer.Optimizer or MixPrecisionOptimizer for main grad.'
    assert level in ['os', 'os_g', 'p_g_os'], 'The level must be os, os_g or p_g_os.'

    def check_dtype(param):
        if False:
            print('Hello World!')
        return param.dtype == paddle.float16
    params_fp16 = list(filter(check_dtype, model.parameters()))
    if scaler is None and len(params_fp16) > 0:
        logger_.warning('the input of scaler is None, please ensure the logic of your scaler outside is same as GroupShardedScaler.')
    if level in ['os', 'os_g']:
        logger_.info('*' * 30)
        logger_.info('Sharded level os uses sharded level os_g achieved now.')
        logger_.info('*' * 30)
        optimizer = GroupShardedOptimizerStage2(params=optimizer._parameter_list, optim=optimizer, group=group, offload=offload, dp_group=dp_group, device=device)
        model = GroupShardedStage2(model, optimizer, group=group, sync_buffers=sync_buffers, buffer_max_size=buffer_max_size, dp_group=dp_group, device=device)
    elif level == 'p_g_os':
        model = GroupShardedStage3(model, optimizer=optimizer, group=group, sync_buffers=sync_buffers, segment_size=segment_size, offload=offload, sync_comm=sync_comm, dp_group=dp_group, device=device, exclude_layer=exclude_layer)
    else:
        raise ValueError('Please enter the correct level.')
    if isinstance(scaler, paddle.amp.GradScaler):
        scaler = GroupShardedScaler(scaler)
    logger_.info('*' * 30)
    logger_.info('If there is a communication hang using group sharded, please check whether the communication operations of each process are unified.')
    logger_.info('*' * 30)
    return (model, optimizer, scaler)

def save_group_sharded_model(model, output, optimizer=None):
    if False:
        while True:
            i = 10
    '\n    Group sharded encapsulated model and optimizer state saving module.\n\n    Note:\n        If using save_group_sharded_model saves the model. When loading again, you need to set the model or optimizer state before using group_sharded_parallel.\n\n    Args:\n        model (Layer): A wrapper for group sharded given model.\n        output (str): Save directory.\n        optimizer (Optimizer, optional): Group sharded encapsulated optimizer. Defaults to None, indicating that the optimizer state is not saved.\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:DISTRIBUTED)\n            >>> import paddle\n            >>> from paddle.nn import Linear\n            >>> from paddle.distributed import fleet\n            >>> from paddle.distributed.sharding import group_sharded_parallel, save_group_sharded_model\n\n            >>> fleet.init(is_collective=True)\n            >>> group = paddle.distributed.new_group([0, 1])\n            >>> model = Linear(1000, 1000)\n\n            >>> clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)\n            >>> optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters(), weight_decay=0.00001, grad_clip=clip)\n\n            >>> # wrap sharding model, optimizer and scaler\n            >>> model, optimizer, scaler = group_sharded_parallel(model, optimizer, "p_g", scaler=scaler)\n\n            >>> img, label = data\n            >>> label.stop_gradient = True\n            >>> img.stop_gradient = True\n\n            >>> out = model(img)\n            >>> loss = paddle.nn.functional.cross_entropy(input=out, label=label)\n\n            >>> loss.backward()\n            >>> optimizer.step()\n            >>> optimizer.clear_grad()\n\n            >>> # save model and optimizer state_dict\n            >>> save_group_sharded_model(model, optimizer, output=output_dir)\n\n    '
    logger_.info('==========Begin to save group sharded model and optimizer==========')
    assert not os.path.isfile(output), f'Saving directory ({output}) should be a directory, not a file'
    os.makedirs(output, exist_ok=True)
    output_model = os.path.join(output, 'model.pdmodel')
    if isinstance(model, GroupShardedStage2):
        paddle.save(model._layer.state_dict(), output_model)
    elif isinstance(model, GroupShardedStage3):
        convert2cpu = True if model._offload else False
        model.get_all_parameters(convert2cpu=convert2cpu)
        paddle.save(model._layer.state_dict(), output_model)
    else:
        raise ValueError('Please use the layer which is wrapped with group_sharded_parallel.')
    if optimizer is not None:
        assert hasattr(optimizer, '_optim'), 'Please use the optimizer which is wrapped with group_sharded_parallel.'
        output_opt = os.path.join(output, 'model.pdopt')
        paddle.save(optimizer._optim.state_dict(), output_opt)
    logger_.info('==========End to save group sharded model and optimizer==========')