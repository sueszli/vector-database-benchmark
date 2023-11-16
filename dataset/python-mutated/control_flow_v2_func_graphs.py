"""FuncGraphs for V2 control flow."""
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops

class ControlFlowFuncGraph(func_graph.FuncGraph):
    """Contains control flow-specific FuncGraph logic."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ControlFlowFuncGraph, self).__init__(*args, **kwargs)
        outer_graph = self.outer_graph
        self._device_function_stack = outer_graph._device_function_stack.copy()
        self.is_control_flow_graph = True
        if ops.executing_eagerly_outside_functions():
            func_graph.override_func_graph_name_scope(self, self.outer_graph.get_name_scope())

class CondBranchFuncGraph(ControlFlowFuncGraph):
    """FuncGraph for branches of tf.cond().

  This is used to distinguish cond branches from other functions.
  """

class WhileCondFuncGraph(ControlFlowFuncGraph):
    """FuncGraph for the condition of tf.while_loop().

  This is used to distinguish while conditions from other functions.
  """

class WhileBodyFuncGraph(ControlFlowFuncGraph):
    """FuncGraph for the body of tf.while_loop().

  This is used to distinguish while bodies from other functions.
  """