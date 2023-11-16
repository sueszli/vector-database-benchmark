import os
import numpy as np
from paddle import base
from paddle.base.framework import _global_flags
from paddle.base.layer_helper import LayerHelper

def check():
    if False:
        while True:
            i = 10
    print("check: _global_flags()['FLAGS_use_mkldnn']=", _global_flags()['FLAGS_use_mkldnn'])
    print("check: base.get_flags('FLAGS_use_mkldnn')=", base.get_flags(['FLAGS_use_mkldnn']))
    print('check: DNNL_VERBOSE=', os.environ['DNNL_VERBOSE'])
    a_np = np.random.uniform(-2, 2, (10, 20, 30)).astype(np.float32)
    helper = LayerHelper(base.unique_name.generate('test'), act='relu')
    func = helper.append_activation
    with base.dygraph.guard(base.core.CPUPlace()):
        a = base.dygraph.to_variable(a_np)
        res1 = func(a)
        res2 = np.maximum(a_np, 0)
    np.testing.assert_array_equal(res1.numpy(), res2)
if __name__ == '__main__':
    try:
        check()
        for (k, v) in sorted(os.environ.items()):
            print(k + ':', v)
        print('\n')
    except Exception as e:
        print(e)
        print(type(e))