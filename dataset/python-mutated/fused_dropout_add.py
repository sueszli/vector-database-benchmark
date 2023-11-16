from paddle import _C_ops
from paddle.base import core
from paddle.common_ops_import import default_main_program
from paddle.framework import LayerHelper, in_dynamic_or_pir_mode

def fused_dropout_add(x, y, p=0.5, training=True, mode='upscale_in_train', name=None):
    if False:
        i = 10
        return i + 15
    '\n    Fused Dropout and Add.\n\n    Args:\n        x (Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64.\n        y (Tensor): The input tensor. The data type is bfloat16, float16, float32 or float64.\n\n        p (float|int, optional): Probability of setting units to zero. Default: 0.5.\n        training (bool, optional): A flag indicating whether it is in train phrase or not. Default: True.\n        mode(str, optional): [\'upscale_in_train\'(default) | \'downscale_in_infer\'].\n\n            1. upscale_in_train (default), upscale the output at training time\n\n                - train: :math:`out = x \\times \\frac{mask}{(1.0 - dropout\\_prob)} + y`\n                - inference: :math:`out = x + y`\n\n            2. downscale_in_infer, downscale the output at inference\n\n                - train: :math:`out = input \\times mask + y`\n                - inference: :math:`out = input \\times (1.0 - dropout\\_prob) + y`\n\n        name (str, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.\n\n    Returns:\n        A Tensor representing the fused dropout and add, has same shape and data type as `x` .\n\n\n    Examples:\n\n        .. code-block:: python\n\n            >>> # doctest: +REQUIRES(env:GPU)\n            >>> import paddle\n            >>> from paddle.incubate.nn.functional import fused_dropout_add\n\n            >>> paddle.set_device(\'gpu\')\n            >>> paddle.seed(2023)\n            >>> x = paddle.randn([4, 10], dtype="float32")\n            >>> y = paddle.randn([4, 10], dtype="float32")\n            >>> out = fused_dropout_add(x, y, p=0.5)\n            >>> print(out)\n            Tensor(shape=[4, 10], dtype=float32, place=Place(gpu:0), stop_gradient=True,\n            [[-0.49133155,  0.53819323, -2.58393312,  0.06336236, -1.09908366,\n               0.22085167,  2.19751787,  0.05034769,  0.53417486,  0.84864247],\n             [ 0.78248203, -1.59652555, -0.14399840, -0.77985179, -0.17006736,\n              -0.30991879, -0.36593807, -0.51025450,  1.46401680,  0.61627960],\n             [ 4.50472546, -0.48472026,  0.60729283,  0.33509624, -0.25593102,\n              -1.45173049,  1.06727099,  0.00440830, -0.77340341,  0.67393088],\n             [ 1.29453969,  0.07568165,  0.71947742, -0.71768606, -2.57172823,\n               1.89179027,  3.26482797,  1.10493207, -1.04569530, -1.04862499]])\n    '
    if isinstance(p, (int, float)):
        if p == 0:
            return x + y
        elif p < 0 or p > 1:
            raise ValueError('p argument should between 0 and 1')
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError("mode argument should be 'downscale_in_infer' or 'upscale_in_train'")
    seed = None
    if in_dynamic_or_pir_mode():
        if default_main_program().random_seed != 0:
            seed = default_main_program().random_seed
        (out, seed_offset) = _C_ops.fused_dropout_add(x, y, None, p, not training, mode, seed if seed is not None else 0, seed is not None)
        return out
    else:
        helper = LayerHelper('fused_dropout_add', **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        seed_offset = helper.create_variable_for_type_inference(dtype=core.VarDesc.VarType.INT64, stop_gradient=True)

        def get_attrs(prog, dropout_prob, is_test, seed):
            if False:
                return 10
            if (seed is None or seed == 0) and prog.random_seed != 0:
                seed = prog.random_seed
            attrs = {'p': dropout_prob, 'is_test': is_test, 'mode': mode, 'seed': seed if seed is not None else 0, 'fix_seed': seed is not None}
            return attrs
        attrs = get_attrs(helper.main_program, p, not training, seed)
        helper.append_op(type='fused_dropout_add', inputs={'x': x, 'y': y, 'seed_tensor': None}, outputs={'out': [out], 'seed_offset': [seed_offset]}, attrs=attrs)
        return out