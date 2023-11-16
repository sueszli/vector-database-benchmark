import unittest
import numpy as np
import paddle
from paddle import _legacy_C_ops
from paddle.base import core
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.base.framework import Variable

def _append_backward_desc(main_program, outs):
    if False:
        i = 10
        return i + 15
    program = main_program.clone()
    targets = []
    for out in outs:
        if isinstance(out, Variable):
            targets.append(program.global_block().var(out.name))
    if targets:
        paddle.base.backward.gradients(targets=targets, inputs=[])
    return program

def _create_out(var):
    if False:
        while True:
            i = 10
    assert isinstance(var, Variable)
    var_desc = var.desc
    out = core.eager.Tensor(var_desc.dtype(), var_desc.shape(), var_desc.name(), var_desc.type(), False)
    return out

@switch_to_static_graph
def _add_build_strategy_for(input_program, start_op_index, end_op_index):
    if False:
        while True:
            i = 10
    compiled_program = paddle.static.CompiledProgram(core.Graph(input_program.desc, start_op_index, end_op_index), build_strategy=paddle.static.BuildStrategy())
    compiled_program._compile(core.Scope(), paddle.framework._current_expected_place())
    ir_graph = paddle.base.framework.IrGraph(compiled_program._graph)
    builded_program = ir_graph.to_program()
    return builded_program

class TestRunProgram(unittest.TestCase):

    def test_eager(self):
        if False:
            while True:
                i = 10
        paddle.set_device('cpu')
        paddle.enable_static()
        x = paddle.static.data(shape=[2, 4], name='x')
        x.stop_gradient = False
        y = paddle.static.data(shape=[4, 2], name='y')
        y.stop_gradient = False
        out = paddle.matmul(x, y)
        main_program = paddle.static.default_main_program()
        program = _append_backward_desc(main_program, [out])
        forward_program = _add_build_strategy_for(program, 0, main_program.desc.block(0).op_size())
        backward_program = _add_build_strategy_for(program, main_program.desc.block(0).op_size() + 1, program.desc.block(0).op_size())
        paddle.disable_static('cpu')
        x_t = paddle.ones([2, 4])
        x_t.name = 'x'
        x_t.stop_gradient = False
        y_t = paddle.ones([4, 2])
        y_t.name = 'y'
        y_t.stop_gradient = False
        fake_var = paddle.zeros([1])
        fake_var.name = 'Fake_var'
        out_t = _create_out(out)
        scope = core.Scope()
        attrs = ['global_block', program.desc.block(0), 'start_op_index', 0, 'end_op_index', main_program.desc.block(0).op_size(), 'is_test', False, 'program_id', paddle.utils._hash_with_id(program), 'param_grad_names', ['Fake_var@GRAD'], 'out_grad_names', [out.name + '@GRAD'], 'x_grad_names', [x_t.name + '@GRAD', y_t.name + '@GRAD'], 'x_names', [x_t.name, y_t.name]]
        use_interpretorcore = True
        attrs.extend(('use_interpretorcore', use_interpretorcore))
        if use_interpretorcore:
            attrs.extend(('forward_global_block', forward_program.desc.block(0), 'backward_global_block', backward_program.desc.block(0)))
        _legacy_C_ops.run_program([x_t, y_t], [fake_var], [out_t], [scope], None, *attrs)
        loss = paddle.mean(out_t)
        loss.backward()
        np.testing.assert_array_equal(np.ones([2, 2]) * 4, out_t.numpy())
        np.testing.assert_array_equal(np.ones([2, 4]) * 0.5, x_t.grad.numpy())
        np.testing.assert_array_equal(np.ones([4, 2]) * 0.5, y_t.grad.numpy())
if __name__ == '__main__':
    unittest.main()