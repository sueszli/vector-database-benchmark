from concurrent.futures import Future
import numpy as np
from megengine.core.ops.builtin import Elemwise
from megengine.core.tensor import megbrain_graph as mgb_graph
from megengine.tensor import Tensor

def test_io():
    if False:
        while True:
            i = 10
    g = mgb_graph.Graph()
    x = Tensor(np.random.randn(3).astype('float32'), device='xpux')._dev_tensor()
    (vx, _) = mgb_graph.input_callback(lambda : x, device=x.comp_node, dtype=x.dtype, graph=g)
    y = Future()
    v = mgb_graph.output_callback(y.set_result, vx)
    f = g.compile(v)
    f()
    np.testing.assert_equal(x.numpy(), y.result().numpy())

def test_io2():
    if False:
        return 10
    g = mgb_graph.Graph()
    g.options.async_exec_level = 4
    (dtype, device) = ('float32', 'xpux')
    px = mgb_graph.InputNode(device=device, dtype=dtype, graph=g)
    py = mgb_graph.OutputNode(px.outputs[0])
    f = g.compile(py.outputs[0])
    for _ in range(3):
        f.execute()
        x = Tensor(np.random.randn(10).astype(dtype), device=device)._dev_tensor()
        px.set_value(x)
        y = py.get_value()
        np.testing.assert_equal(x.numpy(), y.numpy())
        f.wait()

def test_attr_output():
    if False:
        return 10
    g = mgb_graph.Graph()
    g.options.async_exec_level = 4
    (dtype, device) = ('float32', 'xpux')
    px = mgb_graph.InputNode(device=device, dtype=dtype, graph=g)
    py = mgb_graph.AttrOutputNode(px.outputs[0])
    f = g.compile(py.outputs[0])
    for shape in [(2,), (3,), (5,)]:
        f.execute()
        x = Tensor(np.random.randn(*shape).astype(dtype), device=device)._dev_tensor()
        px.set_value(x)
        ay = py.get_value()
        assert ay.shape == shape
        assert ay.dtype == np.dtype(dtype)
        assert ay.device == device
        f.wait()

def test_op():
    if False:
        i = 10
        return i + 15
    g = mgb_graph.Graph()
    x = Tensor(np.random.randn(10).astype('float32'), device='xpux')._dev_tensor()
    (v, _) = mgb_graph.input_callback(lambda : x, device=x.comp_node, dtype=x.dtype, graph=g)
    neg = Elemwise(Elemwise.Mode.NEGATE)
    v = mgb_graph.apply_normal_varnode(neg, v)[0]
    y = Future()
    v = mgb_graph.output_callback(y.set_result, v)
    f = g.compile(v)
    f()
    np.testing.assert_equal(x.numpy(), -y.result().numpy())

def test_exception():
    if False:
        return 10
    err_msg = 'QwQ'

    def throw_exc():
        if False:
            return 10
        raise RuntimeError(err_msg)
    g = mgb_graph.Graph()
    (x, _) = mgb_graph.input_callback(throw_exc, device='xpux', dtype='float32', graph=g)
    neg = Elemwise(Elemwise.Mode.NEGATE)
    y = mgb_graph.OutputNode(mgb_graph.apply_normal_varnode(neg, x)[0])
    f = g.compile(y.outputs[0])
    try:
        f.execute()
        y.get_value()
    except Exception as exc:
        assert err_msg in str(exc)