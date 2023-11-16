from paddle import _C_ops
from paddle.base import core
from paddle.base.data_feeder import check_type, check_variable_and_dtype
from paddle.base.framework import Variable, in_dygraph_mode
from paddle.base.layer_helper import LayerHelper

def check_finite_and_unscale(x, scale, name=None, float_status=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if input X contains all finite data, if yes, scale it by input Scale.\n\n    $$Out = X / scale$$\n\n    If any tensor in X contains Inf or Nan, the Out will generate a indicator.\n    FoundInfinite will be 1 (True), and Out will not be scaled. In this case, the data of\n    Out should not be used, and its data may not be deterministic.\n    Otherwise, FoundInfinite will be 0 (False).\n\n    Args:\n        x(list|tuple): The input tensors of check_finite_and_unscale operator.\n        scale: The scale of check_finite_and_unscale operator.\n        float_status(Tensor): (Only used on NPU) The float status to check overflow.\n    '
    helper = LayerHelper('check_finite_and_unscale', **locals())
    found_inf = helper.create_variable_for_type_inference(dtype='bool')
    if in_dygraph_mode():
        (x, found_inf) = _C_ops.check_finite_and_unscale_(x, scale)
        return (x, found_inf)
    check_type(x, 'x', (tuple, list), 'check_finite_and_unscale')
    for e in x:
        check_variable_and_dtype(e, 'x', ['float16', 'float32', 'float64', 'uint16'], 'check_finite_and_unscale')
    inputs = {'X': x, 'Scale': scale}
    outputs = {'Out': x, 'FoundInfinite': found_inf}
    helper.append_op(type='check_finite_and_unscale', inputs=inputs, outputs=outputs)
    return (x, found_inf)

def update_loss_scaling(x, found_inf, prev_loss_scaling, num_good_steps, num_bad_steps, incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio, stop_update=False, name=None):
    if False:
        return 10
    '\n    Update loss scaling according to overall gradients. If all gradients is\n    finite after incr_every_n_steps, loss scaling will increase by incr_ratio.\n    Otherwise, loss scaling will decrease by decr_ratio after\n    decr_every_n_nan_or_inf steps and each step some gradients are infinite.\n\n    Args:\n        x(list|tuple): The input tensors of update_loss_scaling operator.\n        found_inf (Variable): A boolean variable indicates whether\n                                     there is any infinite gradient.\n        prev_loss_scaling (Variable): Previous loss scaling.\n        num_good_steps (Variable): A variable accumulates good steps in which\n                                   all gradients are finite.\n        num_bad_steps (Variable): A variable accumulates bad steps in which\n                                  some gradients are infinite.\n        incr_every_n_steps (int): A variable represents increasing loss\n                                       scaling every n consecutive steps with\n                                       finite gradients.\n        decr_every_n_nan_or_inf (int): A variable represents decreasing\n                                            loss scaling every n accumulated\n                                            steps with nan or inf gradients.\n        incr_ratio(float): The multiplier to use when increasing the loss\n                           scaling.\n        decr_ratio(float): The less-than-one-multiplier to use when decreasing\n                           loss scaling.\n    '
    if in_dygraph_mode():
        _C_ops.update_loss_scaling_(x, found_inf, prev_loss_scaling, num_good_steps, num_bad_steps, incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio, stop_update)
        return x
    check_variable_and_dtype(prev_loss_scaling, 'prev_loss_scaling', ['float32', 'float64'], 'update_loss_scaling')
    check_type(x, 'x', (tuple, list), 'update_loss_scaling')
    for e in x:
        check_variable_and_dtype(e, 'x', ['float16', 'float32', 'float64', 'uint16'], 'update_loss_scaling')
        if e.dtype in [core.VarDesc.VarType.FP16, core.VarDesc.VarType.BF16]:
            assert prev_loss_scaling.dtype == core.VarDesc.VarType.FP32, 'The dtype of prev_loss_scaling should be float32 when the dtype of x is float16 or bfloat16.'
        else:
            assert prev_loss_scaling.dtype == e.dtype, 'The dtype of prev_loss_scaling should be equal to the dtype of x.'
    helper = LayerHelper('update_loss_scaling', **locals())
    inputs = {'X': x, 'FoundInfinite': found_inf, 'PrevLossScaling': prev_loss_scaling, 'InGoodSteps': num_good_steps, 'InBadSteps': num_bad_steps}
    outputs = {'Out': x, 'LossScaling': prev_loss_scaling, 'OutGoodSteps': num_good_steps, 'OutBadSteps': num_bad_steps}
    attrs = {'incr_every_n_steps': incr_every_n_steps, 'decr_every_n_nan_or_inf': decr_every_n_nan_or_inf, 'incr_ratio': incr_ratio, 'decr_ratio': decr_ratio}
    if isinstance(stop_update, Variable):
        inputs['StopUpdate'] = stop_update
    else:
        attrs['stop_update'] = stop_update
    helper.append_op(type='update_loss_scaling', inputs=inputs, outputs=outputs, attrs=attrs)
    return x