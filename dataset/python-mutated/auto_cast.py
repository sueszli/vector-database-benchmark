import copy
import warnings
import paddle
from paddle.base import core
from paddle.base.framework import _dygraph_tracer, dygraph_only
from paddle.base.wrapped_decorator import signature_safe_contextmanager
from .amp_lists import black_list, white_list
AMP_RELATED_FLAGS = ['FLAGS_cudnn_exhaustive_search', 'FLAGS_conv_workspace_size_limit', 'FLAGS_cudnn_batchnorm_spatial_persistent']
AMP_RELATED_FLAGS_SETTING = {'FLAGS_cudnn_exhaustive_search': 1, 'FLAGS_conv_workspace_size_limit': 1000, 'FLAGS_cudnn_batchnorm_spatial_persistent': 1}
AMP_LEVEL = core.AmpLevel
_g_amp_state_ = None

def amp_state():
    if False:
        return 10
    global _g_amp_state_
    return _g_amp_state_

class AMPGlobalState:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.model_parameters = []
        self.use_master_grad = False
        self.already_register_final_backward_hook = False
        self.amp_dtype = 'float32'

    def __setattr__(self, name, val):
        if False:
            print('Hello World!')
        self.__dict__[name] = val
_amp_global_state = AMPGlobalState()

def amp_global_state():
    if False:
        return 10
    return _amp_global_state

def _update_list(custom_white_list, custom_black_list, level='O1', dtype='float16'):
    if False:
        i = 10
        return i + 15
    "\n    Update black and white list according to users' custom list.\n    "
    if level == 'O0':
        _white_list = set()
        _black_list = set()
        return (_white_list, _black_list)
    _white_list = copy.copy(white_list()[dtype][level])
    _black_list = copy.copy(black_list()[dtype][level])
    if custom_white_list and custom_black_list:
        for op_name in custom_white_list:
            if op_name in custom_black_list:
                raise ValueError('Custom white list overlap custom black list')
    if custom_white_list:
        for op_name in custom_white_list:
            if op_name in _black_list:
                _black_list.remove(op_name)
            _white_list.add(op_name)
    if custom_black_list:
        for op_name in custom_black_list:
            if op_name in _white_list:
                _white_list.remove(op_name)
            _black_list.add(op_name)
    return (_white_list, _black_list)

def _in_amp_guard():
    if False:
        for i in range(10):
            print('nop')
    '\n    Judge whether current code block is in `amp_guard` context.\n    '
    tracer = _dygraph_tracer()
    if tracer:
        if tracer._amp_level == core.AmpLevel.O1:
            return True
        else:
            return False
    else:
        return False

def _in_pure_fp16_guard():
    if False:
        return 10
    tracer = _dygraph_tracer()
    return tracer and tracer._amp_level == core.AmpLevel.O2

def _is_gpu_float16_supported():
    if False:
        i = 10
        return i + 15
    '\n    Judge whether current gpu support float16 amp.\n    '
    prop = paddle.device.cuda.get_device_capability()
    return prop[0] >= 7

def _is_gpu_bfloat16_supported():
    if False:
        for i in range(10):
            print('nop')
    '\n    Judge whether current gpu support bfloat16 amp.\n    '
    prop = paddle.device.cuda.get_device_capability()
    cuda_version = paddle.version.cuda()
    if cuda_version is not None and cuda_version != 'False':
        cuda_version_check = int(cuda_version.split('.')[0]) >= 11
    else:
        cuda_version_check = False
    return prop[0] >= 8 and cuda_version_check

def need_keep_fp32(layer, dtype):
    if False:
        print('Hello World!')
    need_keep_fp32 = False
    if not layer._cast_to_low_precison:
        need_keep_fp32 = True
    elif isinstance(layer, (paddle.nn.BatchNorm, paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D, paddle.nn.SyncBatchNorm)):
        need_keep_fp32 = True
    elif layer._dtype == 'float16' or (dtype == 'float16' and isinstance(layer, (paddle.nn.LayerNorm, paddle.nn.InstanceNorm1D, paddle.nn.InstanceNorm2D, paddle.nn.InstanceNorm3D))):
        need_keep_fp32 = True
    return need_keep_fp32

def set_excluded_layers(models, excluded_layers):
    if False:
        while True:
            i = 10
    excluded_layers_instances = []
    excluded_layers_types = []
    error_message = 'excluded_layers must be either a nn.Layer instance/type or a list of nn.Layer instances/types.'
    if excluded_layers is None:
        excluded_layers = []
    elif isinstance(excluded_layers, paddle.nn.Layer):
        excluded_layers_instances = [excluded_layers]
    elif isinstance(excluded_layers, type) and issubclass(excluded_layers, paddle.nn.Layer):
        excluded_layers_types = [excluded_layers]
    elif isinstance(excluded_layers, list):
        for item in excluded_layers:
            if isinstance(item, paddle.nn.Layer):
                excluded_layers_instances.append(item)
            elif issubclass(item, paddle.nn.Layer):
                excluded_layers_types.append(item)
            else:
                raise TypeError(error_message)
    else:
        raise TypeError(error_message)
    for idx in range(len(excluded_layers_instances)):
        for layer in excluded_layers_instances[idx].sublayers(include_self=True):
            layer._cast_to_low_precison = False
    excluded_layers_types = tuple(excluded_layers_types)
    for idx in range(len(models)):
        for layer in models[idx].sublayers(include_self=True):
            if isinstance(layer, excluded_layers_types):
                layer._cast_to_low_precison = False

@dygraph_only
def amp_initialize(models, dtype, excluded_layers):
    if False:
        print('Hello World!')
    set_excluded_layers(models, excluded_layers)
    for idx in range(len(models)):
        for layer in models[idx].sublayers(include_self=True):
            if need_keep_fp32(layer, dtype):
                continue
            if dtype == 'float16' and isinstance(layer, (paddle.incubate.nn.FusedFeedForward, paddle.incubate.nn.FusedMultiHeadAttention)):
                layer._amp_decorate(dtype=dtype)
                continue
            layer._to_impl(dtype=dtype, include_sublayers=False, floating_only=True)
    return models

def check_models(models):
    if False:
        while True:
            i = 10
    for model in models:
        if not isinstance(model, paddle.nn.Layer):
            raise RuntimeError('Current train mode is pure fp16, models should be paddle.nn.Layer, but receive {}.'.format(type(model)))
        if isinstance(model, paddle.DataParallel):
            raise RuntimeError('For distributed AMP training, you should first use paddle.amp.decorate() to decotate origin model, and then call paddle.DataParallel get distributed model.')

def _is_valid_optimizer(optimizer):
    if False:
        i = 10
        return i + 15
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import DygraphShardingOptimizer, DygraphShardingOptimizerV2
    return isinstance(optimizer, (paddle.optimizer.Optimizer, DygraphShardingOptimizer, DygraphShardingOptimizerV2))

def check_optimizers(optimizers):
    if False:
        for i in range(10):
            print('nop')
    for optimizer in optimizers:
        if not _is_valid_optimizer(optimizer):
            raise RuntimeError('Current train mode is pure fp16, optimizers should be paddle.optimizer.Optimizer or DygraphShardingOptimizer, but receive {}.'.format(type(optimizer)))

@signature_safe_contextmanager
@dygraph_only
def amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level='O1', dtype='float16', use_promote=True):
    if False:
        while True:
            i = 10
    '\n    Create a context which enables auto-mixed-precision(AMP) of operators executed in dynamic graph mode.\n    If enabled, the input data type (float32 or float16) of each operator is decided\n    by autocast algorithm for better performance.\n\n    Commonly, it is used together with `GradScaler` to achieve Auto-Mixed-Precision in\n    imperative mode. It is used together with `decorator` to achieve Pure fp16 in imperative mode.\n\n    Args:\n        enable(bool, optional): Enable auto-mixed-precision or not. Default is True.\n        custom_white_list(set|list|tuple, optional): The custom white_list. It\'s the set of ops that support\n             fp16 calculation and are considered numerically-safe and performance-critical. These ops\n             will be converted to fp16.\n        custom_black_list(set|list|tuple, optional): The custom black_list. The set of ops that support fp16\n             calculation and are considered numerically-dangerous and whose effects may also be\n             observed in downstream ops. These ops will not be converted to fp16.\n        level(str, optional): Auto mixed precision level. Accepted values are "O1" and "O2": O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list;\n             O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don\'t support fp16 kernel and batchnorm. Default is O1(amp)\n        dtype(str, optional): Whether to use \'float16\' or \'bfloat16\'. Default is \'float16\'.\n\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n\n            >>> data = paddle.uniform([10, 3, 32, 32], paddle.float32, -1, 1)\n            >>> conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)\n            >>> conv2d = paddle.amp.amp_decorate(models=conv2d, level=\'O2\')\n            >>> with paddle.amp.amp_guard():\n            ...     conv = conv2d(data)\n            ...     print(conv.dtype)\n            >>> # doctest: +SKIP("This has diff in xdoctest env")\n            paddle.float16\n            >>> # doctest: -SKIP\n            ...\n            >>> with paddle.amp.amp_guard(enable=False):\n            ...     conv = conv2d(data)\n            ...     print(conv.dtype)\n            >>> # doctest: +SKIP("This has diff in xdoctest env")\n            paddle.float32\n            >>> # doctest: -SKIP\n    '
    amp_state = locals()
    global _g_amp_state_
    original_state = _g_amp_state_
    _g_amp_state_ = amp_state
    level = level.upper()
    if not level in ['O0', 'OD', 'O1', 'O2']:
        raise ValueError('level should be O0, OD, O1 or O2.')
    dtype = dtype.lower()
    if enable:
        if not dtype in ['float16', 'bfloat16']:
            raise ValueError("If enable amp, dtype should be 'float16' or 'bfloat16'.")
    tracer = _dygraph_tracer()
    if not tracer:
        raise ValueError('current_tracer is None, maybe it is not in imperative mode.')
    if enable and (not (tracer._expected_place.is_gpu_place() or tracer._expected_place.is_xpu_place() or tracer._expected_place.is_custom_place())):
        warnings.warn('amp_guard can only be enabled on CUDAPlace, XPUPlace, and CustomPlace, current place is %s, so it makes no effect.' % tracer._expected_place)
        enable = False
    if enable:
        if tracer._expected_place.is_xpu_place() and dtype == 'bfloat16':
            warnings.warn('XPUPlace only support float16 amp.')
            enable = False
        if tracer._expected_place.is_custom_place() and dtype == 'bfloat16':
            warnings.warn('CustomPlace only support float16 amp.')
            enable = False
        if tracer._expected_place.is_gpu_place():
            if dtype == 'float16' and (not _is_gpu_float16_supported()):
                prop = paddle.device.cuda.get_device_capability()
                warnings.warn('For float16, amp only support NVIDIA GPU with Compute Capability 7.0 or higher, current GPU is: %s, with Compute Capability: %d.%d.' % (paddle.device.cuda.get_device_name(), prop[0], prop[1]))
                enable = False
            elif dtype == 'bfloat16' and (not _is_gpu_bfloat16_supported()):
                prop = paddle.device.cuda.get_device_capability()
                cuda_version = paddle.version.cuda()
                warnings.warn('For bfloat16, amp only support NVIDIA GPU with Compute Capability 8.0 or higher and CUDA Version 11.0 or higher, current GPU is: %s, with Compute Capability: %d.%d, current CUDA Version is: %s.' % (paddle.device.cuda.get_device_name(), prop[0], prop[1], cuda_version))
                enable = False
    amp_dtype = dtype
    amp_global_state().amp_dtype = amp_dtype
    if level == 'OD':
        amp_level = AMP_LEVEL.OD
    elif level == 'O1':
        amp_level = AMP_LEVEL.O1
    elif level == 'O2':
        amp_level = AMP_LEVEL.O2
    elif level == 'O0':
        amp_level = AMP_LEVEL.O0
    (_white_list, _black_list) = _update_list(custom_white_list, custom_black_list, level, dtype)
    if not enable:
        amp_level = AMP_LEVEL.O0
        amp_dtype = 'float32'
    if amp_global_state().use_master_grad and (not amp_global_state().already_register_final_backward_hook):

        def master_grad_hook():
            if False:
                i = 10
                return i + 15
            core.eager.set_master_grads(amp_global_state().model_parameters)
            amp_global_state().already_register_final_backward_hook = False
        core.eager._add_backward_final_hook(master_grad_hook)
        amp_global_state().already_register_final_backward_hook = True
    if tracer:
        original_amp_level = tracer._amp_level
        tracer._amp_level = amp_level
        (original_white_list, original_black_list) = tracer._get_amp_op_list()
        tracer._set_amp_op_list(_white_list, _black_list)
        original_amp_dtype = tracer._amp_dtype
        tracer._amp_dtype = amp_dtype
        if amp_level == AMP_LEVEL.O2:
            original_use_promote = tracer._use_promote
            tracer._use_promote = use_promote
    try:
        yield
    finally:
        if tracer:
            _g_amp_state_ = original_state
            tracer._amp_level = original_amp_level
            tracer._set_amp_op_list(original_white_list, original_black_list)
            tracer._amp_dtype = original_amp_dtype
            if amp_level == AMP_LEVEL.O2:
                tracer._use_promote = original_use_promote

class StateDictHook:

    def __init__(self, save_dtype):
        if False:
            while True:
                i = 10
        self._save_dtype = save_dtype

    def __call__(self, state_dict):
        if False:
            i = 10
            return i + 15
        for key in state_dict:
            param = state_dict[key]
            if paddle.is_floating_point(param):
                param_applied = paddle.cast(param, self._save_dtype)
                param_applied.name = param.name
                state_dict[key] = param_applied

def _set_multi_precision(optimizer, multi_precision):
    if False:
        return 10
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import DygraphShardingOptimizer, DygraphShardingOptimizerV2
    optimizer = optimizer._inner_opt if isinstance(optimizer, (DygraphShardingOptimizer, DygraphShardingOptimizerV2)) else optimizer
    if hasattr(optimizer, '_multi_precision'):
        optimizer._multi_precision = multi_precision

@dygraph_only
def amp_decorate(models, optimizers=None, level='O1', dtype='float16', master_weight=None, save_dtype=None, master_grad=False, excluded_layers=None):
    if False:
        print('Hello World!')
    '\n    Decorate models and optimizers for auto-mixed-precision. When level is O1(amp), the decorate will do nothing.\n    When level is O2(pure fp16), the decorate will cast all parameters of models to FP16, except BatchNorm, InstanceNorm and LayerNorm.\n\n    Commonly, it is used together with `amp_guard` to achieve Pure fp16 in imperative mode.\n\n    Args:\n        models(Layer|list of Layer, optional): The defined models by user, models must be either a single model or a list of models. Default is None.\n        optimizers(Optimizer|list of Optimizer, optional): The defined optimizers by user, optimizers must be either a single optimizer or a list of optimizers. Default is None.\n        level(str, optional): Auto mixed precision level. Accepted values are "O1" and "O2": O1 represent mixed precision, the decorator will do nothing;\n             O2 represent Pure fp16/bf16, the decorator will cast all parameters of models to FP16/BF16, except BatchNorm, InstanceNorm and LayerNorm. Default is O1(amp)\n        dtype(str, optional): Whether to use \'float16\' or \'bfloat16\'. Default is \'float16\'.\n        master_weight(bool, optinal): For level=\'O2\', whether to use multi-precision during weight updating. If master_weight is None, in O2 level optimizer will use multi-precision. Default is None.\n        save_dtype(float, optional): The save model parameter dtype when use `paddle.save` or `paddle.jit.save`,it should be float16, bfloat16, float32, float64 or None.\n             The save_dtype will not change model parameters dtype, it just change the state_dict dtype. When save_dtype is None, the save dtype is same as model dtype. Default is None.\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> # Demo1: single model and optimizer:\n            >>> import paddle\n            >>> paddle.device.set_device(\'gpu\')\n\n            >>> model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)\n            >>> optimizer = paddle.optimizer.SGD(parameters=model.parameters())\n\n            >>> model, optimizer = paddle.amp.amp_decorate(models=model, optimizers=optimizer, level=\'O2\')\n\n            >>> data = paddle.rand([10, 3, 32, 32])\n\n            >>> with paddle.amp.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level=\'O2\'):\n            ...     output = model(data)\n            ...     print(output.dtype)\n            paddle.float16\n\n            >>> # Demo2: multi models and optimizers:\n            >>> model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)\n            >>> optimizer2 = paddle.optimizer.Adam(parameters=model2.parameters())\n\n            >>> models, optimizers = paddle.amp.amp_decorate(models=[model, model2], optimizers=[optimizer, optimizer2], level=\'O2\')\n\n            >>> data = paddle.rand([10, 3, 32, 32])\n\n            >>> with paddle.amp.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level=\'O2\'):\n            ...     output = models[0](data)\n            ...     output2 = models[1](data)\n            ...     print(output.dtype)\n            ...     print(output2.dtype)\n            paddle.float16\n            paddle.float16\n\n            >>> # Demo3: optimizers is None:\n            >>> model3 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)\n            >>> optimizer3 = paddle.optimizer.Adam(parameters=model2.parameters())\n\n            >>> model = paddle.amp.amp_decorate(models=model3, level=\'O2\')\n\n            >>> data = paddle.rand([10, 3, 32, 32])\n\n            >>> with paddle.amp.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level=\'O2\'):\n            ...     output = model(data)\n            ...     print(output.dtype)\n            paddle.float16\n    '
    if not level in ['O1', 'O2']:
        raise ValueError('level should be O1 or O2, O1 represent AMP train mode, O2 represent Pure fp16 train mode.')
    if not dtype in ['float16', 'bfloat16']:
        raise ValueError('dtype only support float16 or bfloat16.')
    if level == 'O1':
        if optimizers is None:
            return models
        else:
            return (models, optimizers)
    tracer = _dygraph_tracer()
    if not tracer:
        raise ValueError('current_tracer is None, maybe it is not in imperative mode.')
    if not (tracer._expected_place.is_gpu_place() or tracer._expected_place.is_xpu_place() or tracer._expected_place.is_custom_place()):
        if optimizers is None:
            return models
        else:
            return (models, optimizers)
    if tracer._expected_place.is_xpu_place() and dtype == 'bfloat16':
        if optimizers is None:
            return models
        else:
            return (models, optimizers)
    if tracer._expected_place.is_custom_place() and dtype == 'bfloat16':
        if optimizers is None:
            return models
        else:
            return (models, optimizers)
    if tracer._expected_place.is_gpu_place():
        if dtype == 'float16' and (not _is_gpu_float16_supported()) or (dtype == 'bfloat16' and (not _is_gpu_bfloat16_supported())):
            if optimizers is None:
                return models
            else:
                return (models, optimizers)
    models_is_list = False
    if isinstance(models, paddle.nn.Layer):
        models_is_list = False
        models = [models]
        check_models(models)
    elif isinstance(models, list):
        check_models(models)
        models_is_list = True
    else:
        raise TypeError('models must be either a single model or a list of models.')
    amp_initialize(models=models, dtype=dtype, excluded_layers=excluded_layers)
    if optimizers is not None:
        optimizers_is_list = False
        if _is_valid_optimizer(optimizers):
            optimizers_is_list = False
            optimizers = [optimizers]
            check_optimizers(optimizers)
        elif isinstance(optimizers, list):
            check_optimizers(optimizers)
            optimizers_is_list = True
        else:
            raise TypeError('optimizers must be either a single optimizer or a list of optimizers.')
        use_multi_precision = not master_weight is False
        for opt in optimizers:
            _set_multi_precision(opt, use_multi_precision)
        if master_grad:
            amp_global_state().use_master_grad = True
            for idx in range(len(models)):
                amp_global_state().model_parameters.extend(models[idx].parameters())
    if save_dtype is not None:
        if not save_dtype in ['float16', 'bfloat16', 'float32', 'float64']:
            raise ValueError('save_dtype can only be float16 float32 or float64, but your input save_dtype is %s.' % save_dtype)
        for idx in range(len(models)):
            for layer in models[idx].sublayers(include_self=True):
                layer.register_state_dict_hook(StateDictHook(save_dtype))
    if models_is_list:
        if optimizers is not None:
            if optimizers_is_list:
                return (models, optimizers)
            else:
                return (models, optimizers[0])
        else:
            return models
    elif optimizers is not None:
        if optimizers_is_list:
            return (models[0], optimizers)
        else:
            return (models[0], optimizers[0])
    else:
        return models[0]

def auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O1', dtype='float16', use_promote=True):
    if False:
        while True:
            i = 10
    '\n    Create a context which enables auto-mixed-precision(AMP) of operators executed in dynamic graph mode.\n    If enabled, the input data type (float32, float16 or bfloat16) of each operator is decided\n    by autocast algorithm for better performance.\n\n    Commonly, it is used together with `GradScaler` and `decorator` to achieve Auto-Mixed-Precision in\n    imperative mode.\n\n    Args:\n        enable(bool, optional): Enable auto-mixed-precision or not. Default is True.\n        custom_white_list(set|list|tuple, optional): A default white list is already set. Usually there is no need to set custom white list.\n             The set of ops should be considered numerically-safe and performance-critical. These ops will be converted to float16/bfloat16.\n        custom_black_list(set|list|tuple, optional): A default black list is already set. You can set a custom black list according to the model.\n             The set of ops are considered numerically-dangerous and whose effects may also be observed in downstream ops. These ops will not be\n             converted to float16/bfloat16.\n        level(str, optional): Auto mixed precision level. Accepted values are "O1", "O2" and "OD": At the O1 level, operators in the white list\n             will use float16/bfloat16 inputs for calculations, and operators in the black list will use float32 inputs for calculations. At the O2\n             level, model\'s parameters will be casted to float16/bfloat16 by using `decorator`, and operators that have all float16/bfloat16 inputs\n             will be converted to float16/bfloat16, and that have any float32 input will be converted to float32. For the OD level, operators in\n             default white list will compute in float16/bfloat16, and the others will compute in float32. Default is O1.\n        dtype(str, optional): Whether to use \'float16\' or \'bfloat16\'. Default is \'float16\'.\n        use_promote(bool, optional): Whether to promotes to fp32 when op has any float32 inputs. It is only supported when amp level is O2. Default is True.\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n\n            >>> conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)\n            >>> data = paddle.rand([10, 3, 32, 32])\n\n            >>> with paddle.amp.auto_cast():\n            ...     conv = conv2d(data)\n            ...     print(conv.dtype)\n            >>> # doctest: +SKIP("This has diff in xdoctest env")\n            paddle.float16\n            >>> # doctest: -SKIP\n\n            >>> with paddle.amp.auto_cast(enable=False):\n            ...     conv = conv2d(data)\n            ...     print(conv.dtype)\n            >>> # doctest: +SKIP("This has diff in xdoctest env")\n            paddle.float32\n            >>> # doctest: -SKIP\n\n            >>> with paddle.amp.auto_cast(custom_black_list={\'conv2d\'}):\n            ...     conv = conv2d(data)\n            ...     print(conv.dtype)\n            >>> # doctest: +SKIP("This has diff in xdoctest env")\n            paddle.float32\n            >>> # doctest: -SKIP\n\n            >>> a = paddle.rand([2, 3])\n            >>> b = paddle.rand([2, 3])\n            >>> with paddle.amp.auto_cast(custom_white_list={\'elementwise_add\'}):\n            ...     c = a + b\n            ...     print(c.dtype)\n            >>> # doctest: +SKIP("This has diff in xdoctest env")\n            paddle.float16\n            >>> # doctest: -SKIP\n\n            >>> with paddle.amp.auto_cast(custom_white_list={\'elementwise_add\'}, level=\'O2\'):\n            ...     d = a + b\n            ...     print(d.dtype)\n            >>> # doctest: +SKIP("This has diff in xdoctest env")\n            paddle.float16\n            >>> # doctest: -SKIP\n\n    '
    return amp_guard(enable, custom_white_list, custom_black_list, level, dtype, use_promote)

def decorate(models, optimizers=None, level='O1', dtype='float16', master_weight=None, save_dtype=None, master_grad=False, excluded_layers=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Decorate models and optimizers for auto-mixed-precision. When level is O1(amp), the decorate will do nothing.\n    When level is O2(pure float16/bfloat16), the decorate will cast all parameters of models to float16/bfloat16, except BatchNorm, InstanceNorm and LayerNorm.\n\n    Commonly, it is used together with `auto_cast` to achieve Pure float16/bfloat16 in imperative mode.\n\n    Args:\n        models(Layer|list of Layer): The defined models by user, models must be either a single model or a list of models. Default is None.\n        optimizers(Optimizer|list of Optimizer, optional): The defined optimizers by user, optimizers must be either a single optimizer or a list of optimizers. Default is None.\n        level(str, optional): Auto mixed precision level. Accepted values are 'O1' and 'O2': O1 represent mixed precision, the decorator will do nothing;\n             O2 represent Pure float16/bfloat16, the decorator will cast all parameters of models to float16/bfloat16, except BatchNorm, InstanceNorm and LayerNorm. Default is O1(amp)\n        dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.\n        master_weight(bool, optinal): For level='O2', whether to use multi-precision during weight updating. If master_weight is None, in O2 level optimizer will use multi-precision. Default is None.\n        save_dtype(float, optional): The save model parameter dtype when use `paddle.save` or `paddle.jit.save`,it should be float16, bfloat16, float32, float64 or None.\n             The save_dtype will not change model parameters dtype, it just change the state_dict dtype. When save_dtype is None, the save dtype is same as model dtype. Default is None.\n        master_grad(bool, optional): For level='O2', whether to use float32 weight gradients for calculations such as gradient clipping, weight decay, and weight updates. If master_grad is enabled, the weight\n             gradients will be float32 dtype after the backpropagation. Default is False, there is only float16 weight gradients.\n        excluded_layers(Layer|list of Layer, optional): Specify the layers not to be decorated. The weights of these layers will always keep float32 when level is O2. `excluded_layers` can be specified as\n             an Layer instance/type or a list of Layer instances/types. Default is None, the weights of the whole model will be casted to float16 or bfloat16.\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> # Demo1: single model and optimizer:\n            >>> import paddle\n            >>> paddle.device.set_device('gpu')\n\n            >>> model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)\n            >>> optimizer = paddle.optimizer.SGD(parameters=model.parameters())\n\n            >>> model, optimizer = paddle.amp.decorate(models=model, optimizers=optimizer, level='O2')\n\n            >>> data = paddle.rand([10, 3, 32, 32])\n\n            >>> with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):\n            ...     output = model(data)\n            ...     print(output.dtype)\n            paddle.float16\n\n            >>> # Demo2: multi models and optimizers:\n            >>> model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)\n            >>> optimizer2 = paddle.optimizer.Adam(parameters=model2.parameters())\n\n            >>> models, optimizers = paddle.amp.decorate(models=[model, model2], optimizers=[optimizer, optimizer2], level='O2')\n\n            >>> data = paddle.rand([10, 3, 32, 32])\n\n            >>> with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):\n            ...     output = models[0](data)\n            ...     output2 = models[1](data)\n            ...     print(output.dtype)\n            ...     print(output2.dtype)\n            paddle.float16\n            paddle.float16\n\n            >>> # Demo3: optimizers is None:\n            >>> model3 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)\n            >>> optimizer3 = paddle.optimizer.Adam(parameters=model3.parameters())\n\n            >>> model = paddle.amp.decorate(models=model3, level='O2')\n\n            >>> data = paddle.rand([10, 3, 32, 32])\n\n            >>> with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):\n            ...     output = model(data)\n            ...     print(output.dtype)\n            paddle.float16\n\n    "
    return amp_decorate(models, optimizers, level, dtype, master_weight, save_dtype, master_grad, excluded_layers)