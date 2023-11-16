import numpy as np
import nvidia.dali as dali
from nvidia.dali.backend_impl import TensorListGPU
from nvidia.dali.pipeline import Pipeline

def random_shape(max_shape, diff=100):
    if False:
        for i in range(10):
            print('nop')
    for s in max_shape:
        assert s > diff
    return np.array([np.random.randint(s - diff, s) for s in max_shape], dtype=np.int32)

def check_coin_flip(device='cpu', batch_size=32, max_shape=[100000.0], p=None, use_shape_like_input=False):
    if False:
        while True:
            i = 10
    pipe = Pipeline(batch_size=batch_size, device_id=0, num_threads=3, seed=123456)
    with pipe:

        def shape_gen_f():
            if False:
                i = 10
                return i + 15
            return random_shape(max_shape)
        shape_arg = None
        inputs = []
        shape_out = None
        if max_shape is not None:
            if use_shape_like_input:
                shape_like_in = dali.fn.external_source(lambda : np.zeros(shape_gen_f()), device=device, batch=False)
                inputs += [shape_like_in]
                shape_out = dali.fn.shapes(shape_like_in)
            else:
                shape_arg = dali.fn.external_source(shape_gen_f, batch=False)
                shape_out = shape_arg
        outputs = [dali.fn.random.coin_flip(*inputs, device=device, probability=p, shape=shape_arg)]
        if shape_out is not None:
            outputs += [shape_out]
        pipe.set_outputs(*outputs)
    pipe.build()
    outputs = pipe.run()
    data_out = outputs[0].as_cpu() if isinstance(outputs[0], TensorListGPU) else outputs[0]
    shapes_out = None
    if max_shape is not None:
        shapes_out = outputs[1].as_cpu() if isinstance(outputs[1], TensorListGPU) else outputs[1]
    p = p if p is not None else 0.5
    for i in range(batch_size):
        data = np.array(data_out[i])
        assert np.logical_or(data == 0, data == 1).all()
        if max_shape is not None:
            sample_shape = np.array(shapes_out[i])
            assert (data.shape == sample_shape).all()
            total = len(data)
            positive = np.count_nonzero(data)
            np.testing.assert_allclose(p, positive / total, atol=0.005)

def test_coin_flip():
    if False:
        return 10
    batch_size = 8
    for device in ['cpu', 'gpu']:
        for (max_shape, use_shape_like_in) in [([100000], False), ([100000], True), (None, False)]:
            for probability in [None, 0.7, 0.5, 0.0, 1.0]:
                yield (check_coin_flip, device, batch_size, max_shape, probability, use_shape_like_in)