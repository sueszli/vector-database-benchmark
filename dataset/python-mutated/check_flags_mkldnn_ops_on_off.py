import os
import numpy as np
import paddle
from paddle import base
from paddle.base.framework import _global_flags
from paddle.base.layer_helper import LayerHelper

def check():
    if False:
        return 10
    print("check: _global_flags()['FLAGS_use_mkldnn']=", _global_flags()['FLAGS_use_mkldnn'])
    print("check: base.get_flags('FLAGS_use_mkldnn')=", base.get_flags(['FLAGS_use_mkldnn']))
    print('check: DNNL_VERBOSE=', os.environ['DNNL_VERBOSE'])
    print('check: FLAGS_tracer_mkldnn_ops_on=', _global_flags()['FLAGS_tracer_mkldnn_ops_on'])
    print('check: FLAGS_tracer_mkldnn_ops_off=', _global_flags()['FLAGS_tracer_mkldnn_ops_off'])
    a_np = np.random.uniform(-2, 2, (10, 20, 30)).astype(np.float32)
    b_np = np.random.uniform(-5, 5, (10, 20, 30)).astype(np.float32)
    helper = LayerHelper(base.unique_name.generate('test'), act='relu')
    func = helper.append_activation
    with base.dygraph.guard(base.core.CPUPlace()):
        a = base.dygraph.to_variable(a_np)
        b = base.dygraph.to_variable(b_np)
        y = paddle.add(x=a, y=b)
        y = paddle.matmul(x=y, y=b, transpose_y=True)
        res1 = func(y)
        np_res = np.add(a_np, b_np)
        np_res = np.matmul(np_res, np.transpose(b_np, (0, 2, 1)))
        np_res = np.maximum(np_res, 0)
    np.testing.assert_allclose(res1.numpy(), np_res, atol=0.001)
if __name__ == '__main__':
    try:
        check()
    except Exception as e:
        print(e)
        print(type(e))