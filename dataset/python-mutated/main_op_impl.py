"""SavedModel main op implementation."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
_DEPRECATION_MSG = 'This API was designed for TensorFlow v1. See https://www.tensorflow.org/guide/migrate for instructions on how to migrate your code to TensorFlow v2.'

@tf_export(v1=['saved_model.main_op.main_op'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def main_op():
    if False:
        while True:
            i = 10
    'Returns a main op to init variables and tables.\n\n  Returns the main op including the group of ops that initializes all\n  variables, initializes local variables and initialize all tables.\n\n  Returns:\n    The set of ops to be run as part of the main op upon the load operation.\n  '
    init = variables.global_variables_initializer()
    init_local = variables.local_variables_initializer()
    init_tables = lookup_ops.tables_initializer()
    return control_flow_ops.group(init, init_local, init_tables)

@tf_export(v1=['saved_model.main_op_with_restore', 'saved_model.main_op.main_op_with_restore'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def main_op_with_restore(restore_op_name):
    if False:
        for i in range(10):
            print('nop')
    'Returns a main op to init variables, tables and restore the graph.\n\n  Returns the main op including the group of ops that initializes all\n  variables, initialize local variables, initialize all tables and the restore\n  op name.\n\n  Args:\n    restore_op_name: Name of the op to use to restore the graph.\n\n  Returns:\n    The set of ops to be run as part of the main op upon the load operation.\n  '
    with ops.control_dependencies([main_op()]):
        main_op_with_restore = control_flow_ops.group(restore_op_name)
    return main_op_with_restore