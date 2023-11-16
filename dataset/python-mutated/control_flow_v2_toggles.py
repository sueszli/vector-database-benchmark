"""API for enabling v2 control flow."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['enable_control_flow_v2'])
def enable_control_flow_v2():
    if False:
        print('Hello World!')
    'Use control flow v2.\n\n  control flow v2 (cfv2) is an improved version of control flow in TensorFlow\n  with support for higher order derivatives. Enabling cfv2 will change the\n  graph/function representation of control flow, e.g., `tf.while_loop` and\n  `tf.cond` will generate functional `While` and `If` ops instead of low-level\n  `Switch`, `Merge` etc. ops. Note: Importing and running graphs exported\n  with old control flow will still be supported.\n\n  Calling tf.enable_control_flow_v2() lets you opt-in to this TensorFlow 2.0\n  feature.\n\n  Note: v2 control flow is always enabled inside of tf.function. Calling this\n  function is not required.\n  '
    logging.vlog(1, 'Enabling control flow v2')
    ops._control_flow_api_gauge.get_cell().set(True)
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

@tf_export(v1=['disable_control_flow_v2'])
def disable_control_flow_v2():
    if False:
        print('Hello World!')
    'Opts out of control flow v2.\n\n  Note: v2 control flow is always enabled inside of tf.function. Calling this\n  function has no effect in that case.\n\n  If your code needs tf.disable_control_flow_v2() to be called to work\n  properly please file a bug.\n  '
    logging.vlog(1, 'Disabling control flow v2')
    ops._control_flow_api_gauge.get_cell().set(False)
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = False

@tf_export(v1=['control_flow_v2_enabled'])
def control_flow_v2_enabled():
    if False:
        return 10
    'Returns `True` if v2 control flow is enabled.\n\n  Note: v2 control flow is always enabled inside of tf.function.\n  '
    return control_flow_util.EnableControlFlowV2(ops.get_default_graph())