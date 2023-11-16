import inspect
import nose
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as dmath
import nvidia.dali.types as types
import os
import random
import re
from functools import partial
from nose.plugins.attrib import attr
from nose.tools import nottest
from nvidia.dali.pipeline import Pipeline, pipeline_def
from nvidia.dali.pipeline.experimental import pipeline_def as experimental_pipeline_def
from nvidia.dali.plugin.numba.fn.experimental import numba_function
import test_utils
from segmentation_test_utils import make_batch_select_masks
from test_detection_pipeline import coco_anchors
from test_optical_flow import load_frames, is_of_supported
from test_utils import module_functions, has_operator, restrict_platform
"\nHow to test variable (iter-to-iter) batch size for a given op?\n-------------------------------------------------------------------------------\nThe idea is to create a Pipeline that assumes i2i variability, run 2 iterations\nand compare them with ad-hoc created Pipelines for given (constant) batch sizes.\nThis can be easily done using `check_batch` function below.\n\nOn top of that, there are some utility functions and routines to help with some\ncommon cases:\n1. If the operator is typically processing image-like data (i.e. 3-dim, uint8,\n   0-255, with shape like [640, 480, 3]) and you want to test default arguments\n   only, please add a record to the `ops_image_default_args` list\n2. If the operator is typically processing image-like data (i.e. 3-dim, uint8,\n   0-255, with shape like [640, 480, 3]) and you want to specify any number of\n   its arguments, please add a record to the `ops_image_custom_args` list\n3. If the operator is typically processing audio-like data (i.e. 1-dim, float,\n   0.-1.) please add a record to the `float_array_ops` list\n4. If the operator supports sequences, please add a record to the\n   `sequence_ops` list\n5. If your operator case doesn't fit any of the above, please create a nosetest\n   function, in which you can define a function, that returns not yet built\n   pipeline, and pass it to the `check_batch` function.\n6. If your operator performs random operation, this approach won't provide\n   a comparable result. In this case, the best thing you can do is to check\n   whether the operator works, without qualitative comparison. Use `run_pipeline`\n   instead of `check_pipeline`.\n"

def generate_data(max_batch_size, n_iter, sample_shape, lo=0.0, hi=1.0, dtype=np.float32):
    if False:
        print('Hello World!')
    '\n    Generates an epoch of data, that will be used for variable batch size verification.\n\n    :param max_batch_size: Actual sizes of every batch in the epoch will be less or equal\n                           to max_batch_size\n    :param n_iter: Number of iterations in the epoch\n    :param sample_shape: If sample_shape is callable, shape of every sample will be determined by\n                         calling sample_shape. In this case, every call to sample_shape has to\n                         return a tuple of integers. If sample_shape is a tuple, this will be a\n                         shape of every sample.\n    :param lo: Begin of the random range\n    :param hi: End of the random range\n    :param dtype: Numpy data type\n    :return: An epoch of data\n    '
    batch_sizes = np.array([max_batch_size // 2, max_batch_size // 4, max_batch_size])
    if isinstance(sample_shape, tuple):

        def sample_shape_wrapper():
            if False:
                i = 10
                return i + 15
            return sample_shape
        size_fn = sample_shape_wrapper
    elif inspect.isfunction(sample_shape):
        size_fn = sample_shape
    else:
        raise RuntimeError('`sample_shape` shall be either a tuple or a callable. Provide `(val,)` tuple for 1D shape')
    if np.issubdtype(dtype, np.integer):
        return [np.random.randint(lo, hi, size=(bs,) + size_fn(), dtype=dtype) for bs in batch_sizes]
    elif np.issubdtype(dtype, np.float32):
        ret = (np.random.random_sample(size=(bs,) + size_fn()) for bs in batch_sizes)
        ret = map(lambda batch: (hi - lo) * batch + lo, ret)
        ret = map(lambda batch: batch.astype(dtype), ret)
        return list(ret)
    elif np.issubdtype(dtype, bool):
        assert isinstance(lo, bool)
        assert isinstance(hi, bool)
        return [np.random.choice(a=[lo, hi], size=(bs,) + size_fn()) for bs in batch_sizes]
    else:
        raise RuntimeError(f'Invalid type argument: {dtype}')

def single_op_pipeline(max_batch_size, input_data, device, *, input_layout=None, operator_fn=None, needs_input=True, **opfn_args):
    if False:
        print('Hello World!')
    pipe = Pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
    with pipe:
        input = fn.external_source(source=input_data, cycle=False, device=device, layout=input_layout)
        if operator_fn is None:
            output = input
        elif needs_input:
            output = operator_fn(input, **opfn_args)
        else:
            output = operator_fn(**opfn_args)
        if needs_input:
            pipe.set_outputs(output)
        else:
            pipe.set_outputs(output, input)
    return pipe

def get_batch_size(batch):
    if False:
        print('Hello World!')
    '\n    Returns the batch size in samples\n\n    :param batch: List of input batches, if there is one input a batch can be either\n                  a numpy array or a list, for multiple inputs it can be tuple of lists or\n                  numpy arrays.\n    '
    if isinstance(batch, tuple):
        return get_batch_size(batch[0])
    elif isinstance(batch, list):
        return len(batch)
    else:
        return batch.shape[0]

def run_pipeline(input_epoch, pipeline_fn, *, devices: list=['cpu', 'gpu'], **pipeline_fn_args):
    if False:
        print('Hello World!')
    "\n    Verifies, if given pipeline supports iter-to-iter variable batch size\n\n    This function verifies only if given pipeline runs without crashing.\n    There is no qualitative verification. Use this for checking pipelines\n    based on random operators (as they can't be verifies against one another).\n\n    :param input_epoch: List of input batches, if there is one input a batch can be either\n                        a numpy array or a list, for multiple inputs it can be tuple of lists or\n                        numpy arrays.\n    :param pipeline_fn: Function, that returns created (but not built) pipeline.\n                        Its signature should be (at least):\n                        pipeline_fn(max_batch_size, input_data, device, ...)\n    :param devices: Devices to run the check on\n    :param pipeline_fn_args: Additional args to pipeline_fn\n    "
    for device in devices:
        n_iter = len(input_epoch)
        max_bs = max((get_batch_size(batch) for batch in input_epoch))
        var_pipe = pipeline_fn(max_bs, input_epoch, device, **pipeline_fn_args)
        var_pipe.build()
        for _ in range(n_iter):
            var_pipe.run()

def check_pipeline(input_epoch, pipeline_fn, *, devices: list=['cpu', 'gpu'], eps=1e-07, **pipeline_fn_args):
    if False:
        i = 10
        return i + 15
    '\n    Verifies, if given pipeline supports iter-to-iter variable batch size\n\n    This function conducts qualitative verification. It compares the result of\n    running multiple iterations of the same pipeline (with possible varying batch sizes,\n    according to `input_epoch`) with results of the ad-hoc created pipelines per iteration\n\n    :param input_epoch: List of input batches, if there is one input a batch can be either\n                        a numpy array or a list, for multiple inputs it can be tuple of lists or\n                        numpy arrays.\n    :param pipeline_fn: Function, that returns created (but not built) pipeline.\n                        Its signature should be (at least):\n                        pipeline_fn(max_batch_size, input_data, device, ...)\n    :param devices: Devices to run the check on\n    :param eps: Epsilon for mean error\n    :param pipeline_fn_args: Additional args to pipeline_fn\n    '
    for device in devices:
        n_iter = len(input_epoch)
        max_bs = max((get_batch_size(batch) for batch in input_epoch))
        var_pipe = pipeline_fn(max_bs, input_epoch, device, **pipeline_fn_args)
        var_pipe.build()
        for iter_idx in range(n_iter):
            iter_input = input_epoch[iter_idx]
            batch_size = get_batch_size(iter_input)
            const_pipe = pipeline_fn(batch_size, [iter_input], device, **pipeline_fn_args)
            const_pipe.build()
            test_utils.compare_pipelines(var_pipe, const_pipe, batch_size=batch_size, N_iterations=1, eps=eps)

def image_like_shape_generator():
    if False:
        print('Hello World!')
    return (random.randint(160, 161), random.randint(80, 81), 3)

def array_1d_shape_generator():
    if False:
        i = 10
        return i + 15
    return (random.randint(300, 400),)

def custom_shape_generator(*args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Fully configurable shape generator.\n    Returns a callable which serves as a non-uniform & random shape generator to generate_epoch\n\n    Usage:\n    custom_shape_generator(dim1_lo, dim1_hi, dim2_lo, dim2_hi, etc...)\n    '
    assert len(args) % 2 == 0, 'Incorrect number of arguments'
    ndims = len(args) // 2
    gen_conf = [[args[2 * i], args[2 * i + 1]] for i in range(ndims)]
    return lambda : tuple([random.randint(lohi[0], lohi[1]) for lohi in gen_conf])

def image_data_helper(operator_fn, opfn_args={}):
    if False:
        for i in range(10):
            print('nop')
    data = generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8)
    check_pipeline(data, pipeline_fn=single_op_pipeline, input_layout='HWC', operator_fn=operator_fn, **opfn_args)

def float_array_helper(operator_fn, opfn_args={}):
    if False:
        print('Hello World!')
    data = generate_data(31, 13, array_1d_shape_generator)
    check_pipeline(data, pipeline_fn=single_op_pipeline, operator_fn=operator_fn, **opfn_args)

def sequence_op_helper(operator_fn, opfn_args={}):
    if False:
        while True:
            i = 10
    data = generate_data(31, 13, custom_shape_generator(3, 7, 160, 200, 80, 100, 3, 3), lo=0, hi=255, dtype=np.uint8)
    check_pipeline(data, pipeline_fn=single_op_pipeline, input_layout='FHWC', operator_fn=operator_fn, **opfn_args)

def random_op_helper(operator_fn, opfn_args={}):
    if False:
        return 10
    run_pipeline(generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8), pipeline_fn=single_op_pipeline, operator_fn=operator_fn, **opfn_args)

def test_external_source():
    if False:
        while True:
            i = 10
    check_pipeline(generate_data(31, 13, custom_shape_generator(2, 4, 2, 4)), single_op_pipeline)
ops_image_default_args = [fn.brightness, fn.brightness_contrast, fn.cat, fn.color_twist, fn.contrast, fn.copy, fn.crop_mirror_normalize, fn.dump_image, fn.hsv, fn.hue, fn.jpeg_compression_distortion, fn.reductions.mean, fn.reductions.mean_square, fn.reductions.rms, fn.reductions.min, fn.reductions.max, fn.reductions.sum, fn.saturation, fn.shapes, fn.sphere, fn.stack, fn.water]

def test_ops_image_default_args():
    if False:
        print('Hello World!')
    for op in ops_image_default_args:
        yield (image_data_helper, op, {})

def numba_set_all_values_to_255_batch(out0, in0):
    if False:
        while True:
            i = 10
    out0[0][:] = 255

def numba_setup_out_shape(out_shape, in_shape):
    if False:
        print('Hello World!')
    out_shape[0] = in_shape[0]
ops_image_custom_args = [(fn.cast, {'dtype': types.INT32}), (fn.color_space_conversion, {'image_type': types.BGR, 'output_type': types.RGB}), (fn.coord_transform, {'M': 0.5, 'T': 2}), (fn.coord_transform, {'T': 2}), (fn.coord_transform, {'M': 0.5}), (fn.crop, {'crop': (5, 5)}), (fn.experimental.equalize, {'devices': ['gpu']}), (fn.erase, {'anchor': [0.3], 'axis_names': 'H', 'normalized_anchor': True, 'shape': [0.1], 'normalized_shape': True}), (fn.fast_resize_crop_mirror, {'crop': [5, 5], 'resize_shorter': 10, 'devices': ['cpu']}), (fn.flip, {'horizontal': True}), (fn.gaussian_blur, {'window_size': 5}), (fn.get_property, {'key': 'layout'}), (fn.laplacian, {'window_size': 3}), (fn.laplacian, {'window_size': 3, 'smoothing_size': 1}), (fn.laplacian, {'window_size': 3, 'normalized_kernel': True}), (fn.normalize, {'batch': True}), (fn.pad, {'fill_value': -1, 'axes': (0,), 'shape': (10,)}), (fn.pad, {'fill_value': -1, 'axes': (0,), 'align': 16}), (fn.paste, {'fill_value': 69, 'ratio': 1, 'devices': ['gpu']}), (fn.per_frame, {'replace': True, 'devices': ['cpu']}), (fn.resize, {'resize_x': 50, 'resize_y': 50}), (fn.resize_crop_mirror, {'crop': [5, 5], 'resize_shorter': 10, 'devices': ['cpu']}), (fn.experimental.tensor_resize, {'sizes': [50, 50], 'axes': [0, 1]}), (fn.rotate, {'angle': 25}), (fn.transpose, {'perm': [2, 0, 1]}), (fn.warp_affine, {'matrix': (0.1, 0.9, 10, 0.8, -0.2, -20)}), (fn.expand_dims, {'axes': 1, 'new_axis_names': 'Z'}), (fn.grid_mask, {'angle': 2.6810782, 'ratio': 0.38158387, 'tile': 51}), (numba_function, {'batch_processing': True, 'devices': ['cpu'], 'in_types': [types.UINT8], 'ins_ndim': [3], 'out_types': [types.UINT8], 'outs_ndim': [3], 'run_fn': numba_set_all_values_to_255_batch, 'setup_fn': numba_setup_out_shape}), (numba_function, {'batch_processing': False, 'devices': ['cpu'], 'in_types': [types.UINT8], 'ins_ndim': [3], 'out_types': [types.UINT8], 'outs_ndim': [3], 'run_fn': numba_set_all_values_to_255_batch, 'setup_fn': numba_setup_out_shape}), (fn.multi_paste, {'in_ids': np.zeros([31], dtype=np.int32), 'output_size': [300, 300, 3]}), (fn.experimental.median_blur, {'devices': ['gpu']})]

def test_ops_image_custom_args():
    if False:
        while True:
            i = 10
    for (op, args) in ops_image_custom_args:
        yield (image_data_helper, op, args)
float_array_ops = [(fn.power_spectrum, {'devices': ['cpu']}), (fn.preemphasis_filter, {}), (fn.spectrogram, {'nfft': 60, 'window_length': 50, 'window_step': 25}), (fn.to_decibels, {}), (fn.audio_resample, {'devices': ['cpu'], 'scale': 1.2})]

def test_float_array_ops():
    if False:
        for i in range(10):
            print('nop')
    for (op, args) in float_array_ops:
        yield (float_array_helper, op, args)
random_ops = [(fn.jitter, {'devices': ['gpu']}), (fn.random_resized_crop, {'size': 69}), (fn.noise.gaussian, {}), (fn.noise.shot, {}), (fn.noise.salt_and_pepper, {}), (fn.segmentation.random_mask_pixel, {'devices': ['cpu']}), (fn.roi_random_crop, {'devices': ['cpu'], 'crop_shape': [10, 15, 3], 'roi_start': [25, 20, 0], 'roi_shape': [40, 30, 3]})]

def test_random_ops():
    if False:
        print('Hello World!')
    for (op, args) in random_ops:
        yield (random_op_helper, op, args)
sequence_ops = [(fn.cast, {'dtype': types.INT32}), (fn.copy, {}), (fn.crop, {'crop': (5, 5)}), (fn.crop_mirror_normalize, {'mirror': 1, 'output_layout': 'FCHW'}), (fn.erase, {'anchor': [0.3], 'axis_names': 'H', 'normalized_anchor': True, 'shape': [0.1], 'normalized_shape': True}), (fn.flip, {'horizontal': True}), (fn.gaussian_blur, {'window_size': 5}), (fn.normalize, {'batch': True}), (fn.per_frame, {'devices': ['cpu']}), (fn.resize, {'resize_x': 50, 'resize_y': 50})]

def test_sequence_ops():
    if False:
        for i in range(10):
            print('nop')
    for (op, args) in sequence_ops:
        yield (sequence_op_helper, op, args)

def test_batch_permute():
    if False:
        print('Hello World!')

    def pipe(max_batch_size, input_data, device):
        if False:
            print('Hello World!')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        perm = fn.batch_permutation(seed=420)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.permute_batch(data, indices=perm)
        pipe.set_outputs(processed)
        return pipe
    run_pipeline(generate_data(31, 13, image_like_shape_generator), pipeline_fn=pipe)

def test_coin_flip():
    if False:
        print('Hello World!')

    def pipe(max_batch_size, input_data, device):
        if False:
            print('Hello World!')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        depthwise = fn.random.coin_flip()
        horizontal = fn.random.coin_flip()
        vertical = fn.random.coin_flip()
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.flip(data, depthwise=depthwise, horizontal=horizontal, vertical=vertical)
        pipe.set_outputs(processed)
        return pipe
    run_pipeline(generate_data(31, 13, image_like_shape_generator), pipeline_fn=pipe, devices=['cpu'])

def test_uniform():
    if False:
        print('Hello World!')

    def pipe(max_batch_size, input_data, device):
        if False:
            return 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        dist = fn.random.uniform()
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = data * dist
        pipe.set_outputs(processed)
        return pipe
    run_pipeline(generate_data(31, 13, array_1d_shape_generator), pipeline_fn=pipe)

def test_random_normal():
    if False:
        return 10

    def pipe_input(max_batch_size, input_data, device):
        if False:
            return 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        dist = fn.random.normal(data)
        pipe.set_outputs(dist)
        return pipe

    def pipe_no_input(max_batch_size, input_data, device):
        if False:
            return 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        dist = data + fn.random.normal()
        pipe.set_outputs(dist)
        return pipe
    run_pipeline(generate_data(31, 13, image_like_shape_generator), pipeline_fn=pipe_input)
    run_pipeline(generate_data(31, 13, image_like_shape_generator), pipeline_fn=pipe_no_input)

def no_input_op_helper(operator_fn, opfn_args={}):
    if False:
        for i in range(10):
            print('nop')
    data = generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8)
    check_pipeline(data, pipeline_fn=single_op_pipeline, input_layout='HWC', operator_fn=operator_fn, needs_input=False, **opfn_args)
no_input_ops = [(fn.constant, {'fdata': 3.1415, 'shape': (10, 10)}), (fn.transforms.translation, {'offset': (2, 3), 'devices': ['cpu']}), (fn.transforms.scale, {'scale': (2, 3), 'devices': ['cpu']}), (fn.transforms.rotation, {'angle': 30.0, 'devices': ['cpu']}), (fn.transforms.shear, {'shear': (2.0, 1.0), 'devices': ['cpu']}), (fn.transforms.crop, {'from_start': (0.0, 1.0), 'from_end': (1.0, 1.0), 'to_start': (0.2, 0.3), 'to_end': (0.8, 0.5), 'devices': ['cpu']})]

def test_no_input_ops():
    if False:
        i = 10
        return i + 15
    for (op, args) in no_input_ops:
        yield (no_input_op_helper, op, args)

def test_combine_transforms():
    if False:
        return 10

    def pipe(max_batch_size, input_data, device):
        if False:
            while True:
                i = 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            batch_size_setter = fn.external_source(source=input_data, cycle=False, device=device)
            t = fn.transforms.translation(offset=(1, 2))
            r = fn.transforms.rotation(angle=30.0)
            s = fn.transforms.scale(scale=(2, 3))
            out = fn.transforms.combine(t, r, s)
        pipe.set_outputs(out, batch_size_setter)
        return pipe
    check_pipeline(generate_data(31, 13, custom_shape_generator(2, 4), lo=1, hi=255, dtype=np.uint8), pipeline_fn=pipe, devices=['cpu'])

@attr('pytorch')
def test_dl_tensor_python_function():
    if False:
        i = 10
        return i + 15
    import torch.utils.dlpack as torch_dlpack

    def dl_tensor_operation(tensor):
        if False:
            print('Hello World!')
        tensor = torch_dlpack.from_dlpack(tensor)
        tensor_n = tensor.double() / 255
        ret = tensor_n.sin()
        ret = torch_dlpack.to_dlpack(ret)
        return ret

    def batch_dl_tensor_operation(tensors):
        if False:
            for i in range(10):
                print('nop')
        out = [dl_tensor_operation(t) for t in tensors]
        return out

    def pipe(max_batch_size, input_data, device, input_layout=None):
        if False:
            print('Hello World!')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0, exec_async=False, exec_pipelined=False)
        with pipe:
            input = fn.external_source(source=input_data, cycle=False, device=device, layout=input_layout)
            output_batch = fn.dl_tensor_python_function(input, function=batch_dl_tensor_operation, batch_processing=True)
            output_sample = fn.dl_tensor_python_function(input, function=dl_tensor_operation, batch_processing=False)
            pipe.set_outputs(output_batch, output_sample, input)
        return pipe
    check_pipeline(generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8), pipeline_fn=pipe, devices=['cpu'])

def test_random_object_bbox():
    if False:
        for i in range(10):
            print('nop')

    def pipe(max_batch_size, input_data, device):
        if False:
            for i in range(10):
                print('nop')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            data = fn.external_source(source=input_data, batch=False, cycle='quiet', device=device)
            out = fn.segmentation.random_object_bbox(data)
        pipe.set_outputs(*out)
        return pipe
    get_data = [np.int32([[1, 0, 0, 0], [1, 2, 2, 1], [1, 1, 2, 0], [2, 0, 0, 1]]), np.int32([[0, 3, 3, 0], [1, 0, 1, 2], [0, 1, 1, 0], [0, 2, 0, 1], [0, 2, 2, 1]])]
    run_pipeline(get_data, pipeline_fn=pipe, devices=['cpu'])

def test_math_ops():
    if False:
        return 10

    def pipe(max_batch_size, input_data, device):
        if False:
            for i in range(10):
                print('nop')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            (data, data2) = fn.external_source(source=input_data, cycle=False, device=device, num_outputs=2)
            processed = [-data, +data, data * data2, data + data2, data - data2, data / data2, data // data2, data ** data2, (data == data2) * 1, (data != data2) * 1, (data < data2) * 1, (data <= data2) * 1, (data > data2) * 1, (data >= data2) * 1, data & data, data | data, data ^ data, dmath.abs(data), dmath.fabs(data), dmath.floor(data), dmath.ceil(data), dmath.pow(data, 2), dmath.fpow(data, 1.5), dmath.min(data, 2), dmath.max(data, 50), dmath.clamp(data, 10, 50), dmath.sqrt(data), dmath.rsqrt(data), dmath.cbrt(data), dmath.exp(data), dmath.log(data), dmath.log2(data), dmath.log10(data), dmath.sin(data), dmath.cos(data), dmath.tan(data), dmath.asin(data), dmath.acos(data), dmath.atan(data), dmath.atan2(data, 3), dmath.sinh(data), dmath.cosh(data), dmath.tanh(data), dmath.asinh(data), dmath.acosh(data), dmath.atanh(data)]
        pipe.set_outputs(*processed)
        return pipe

    def get_data(batch_size):
        if False:
            while True:
                i = 10
        test_data_shape = [random.randint(5, 21), random.randint(5, 21), 3]
        data1 = [np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        data2 = [np.random.randint(1, 4, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        return (data1, data2)
    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    check_pipeline(input_data, pipeline_fn=pipe)

def test_squeeze_op():
    if False:
        while True:
            i = 10

    def pipe(max_batch_size, input_data, device, input_layout=None):
        if False:
            while True:
                i = 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            data = fn.external_source(source=input_data, cycle=False, device=device, layout=input_layout)
            out = fn.expand_dims(data, axes=[0, 2], new_axis_names='YZ')
            out = fn.squeeze(out, axis_names='Z')
        pipe.set_outputs(out)
        return pipe
    check_pipeline(generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8), pipeline_fn=pipe, input_layout='HWC')

def test_box_encoder_op():
    if False:
        while True:
            i = 10

    def pipe(max_batch_size, input_data, device, input_layout=None):
        if False:
            for i in range(10):
                print('nop')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            (boxes, lables) = fn.external_source(device=device, source=input_data, num_outputs=2)
            (processed, _) = fn.box_encoder(boxes, lables, anchors=coco_anchors())
        pipe.set_outputs(processed)
        return pipe

    def get_data(batch_size):
        if False:
            i = 10
            return i + 15
        obj_num = random.randint(1, 20)
        test_box_shape = [obj_num, 4]
        test_lables_shape = [obj_num, 1]
        bboxes = [np.random.random(size=test_box_shape).astype(dtype=np.float32) for _ in range(batch_size)]
        labels = [np.random.randint(0, 255, size=test_lables_shape, dtype=np.int32) for _ in range(batch_size)]
        return (bboxes, labels)
    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    check_pipeline(input_data, pipeline_fn=pipe, devices=['cpu'])

def test_remap():
    if False:
        print('Hello World!')

    def pipe(max_batch_size, input_data, device):
        if False:
            i = 10
            return i + 15
        pipe = Pipeline(batch_size=max_batch_size, device_id=0, num_threads=4)
        with pipe:
            (input, mapx, mapy) = fn.external_source(device=device, source=input_data, num_outputs=3)
            out = fn.experimental.remap(input, mapx, mapy)
        pipe.set_outputs(out)
        return pipe

    def get_data(batch_size):
        if False:
            while True:
                i = 10
        input_shape = [480, 640, 3]
        mapx_shape = mapy_shape = [480, 640]
        input = [np.random.randint(0, 255, size=input_shape, dtype=np.uint8) for _ in range(batch_size)]
        mapx = [640 * np.random.random(size=mapx_shape).astype(np.float32) for _ in range(batch_size)]
        mapy = [480 * np.random.random(size=mapy_shape).astype(np.float32) for _ in range(batch_size)]
        return (input, mapx, mapy)
    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    check_pipeline(input_data, pipeline_fn=pipe, devices=['gpu'])

def test_random_bbox_crop_op():
    if False:
        i = 10
        return i + 15

    def pipe(max_batch_size, input_data, device):
        if False:
            i = 10
            return i + 15
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            (boxes, lables) = fn.external_source(device=device, source=input_data, num_outputs=2)
            processed = fn.random_bbox_crop(boxes, lables, aspect_ratio=[0.5, 2.0], thresholds=[0.1, 0.3, 0.5], scaling=[0.8, 1.0], bbox_layout='xyXY')
        pipe.set_outputs(*processed)
        return pipe

    def get_data(batch_size):
        if False:
            i = 10
            return i + 15
        obj_num = random.randint(1, 20)
        test_box_shape = [obj_num, 4]
        test_lables_shape = [obj_num, 1]
        bboxes = [np.random.random(size=test_box_shape).astype(dtype=np.float32) for _ in range(batch_size)]
        labels = [np.random.randint(0, 255, size=test_lables_shape, dtype=np.int32) for _ in range(batch_size)]
        return (bboxes, labels)
    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    run_pipeline(input_data, pipeline_fn=pipe, devices=['cpu'])

def test_ssd_random_crop_op():
    if False:
        print('Hello World!')

    def pipe(max_batch_size, input_data, device):
        if False:
            return 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            (data, boxes, lables) = fn.external_source(device=device, source=input_data, num_outputs=3)
            processed = fn.ssd_random_crop(data, boxes, lables)
        pipe.set_outputs(*processed)
        return pipe

    def get_data(batch_size):
        if False:
            for i in range(10):
                print('nop')
        obj_num = random.randint(1, 20)
        test_data_shape = [50, 20, 3]
        test_box_shape = [obj_num, 4]
        test_lables_shape = [obj_num]
        data = [np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        bboxes = [np.random.random(size=test_box_shape).astype(dtype=np.float32) for _ in range(batch_size)]
        labels = [np.random.randint(0, 255, size=test_lables_shape, dtype=np.int32) for _ in range(batch_size)]
        return (data, bboxes, labels)
    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    run_pipeline(input_data, pipeline_fn=pipe, devices=['cpu'])

def test_reshape():
    if False:
        while True:
            i = 10
    data = generate_data(31, 13, (160, 80, 3), lo=0, hi=255, dtype=np.uint8)
    check_pipeline(data, pipeline_fn=single_op_pipeline, operator_fn=fn.reshape, shape=(160 / 2, 80 * 2, 3))

def test_slice():
    if False:
        return 10

    def pipe(max_batch_size, input_data, device):
        if False:
            return 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.slice(data, 0.1, 0.5, axes=0, device=device)
        pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, image_like_shape_generator, lo=0, hi=255, dtype=np.uint8), pipeline_fn=pipe)

def test_bb_flip():
    if False:
        while True:
            i = 10
    check_pipeline(generate_data(31, 13, custom_shape_generator(150, 250, 4, 4)), single_op_pipeline, operator_fn=fn.bb_flip)

def test_1_hot():
    if False:
        return 10
    data = generate_data(31, 13, array_1d_shape_generator, lo=0, hi=255, dtype=np.uint8)
    check_pipeline(data, single_op_pipeline, operator_fn=fn.one_hot)

def test_bbox_paste():
    if False:
        while True:
            i = 10

    def pipe(max_batch_size, input_data, device):
        if False:
            print('Hello World!')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        paste_posx = fn.random.uniform(range=(0, 1))
        paste_posy = fn.random.uniform(range=(0, 1))
        paste_ratio = fn.random.uniform(range=(1, 2))
        processed = fn.bbox_paste(data, paste_x=paste_posx, paste_y=paste_posy, ratio=paste_ratio)
        pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, custom_shape_generator(150, 250, 4, 4)), pipe, eps=0.5, devices=['cpu'])

def test_coord_flip():
    if False:
        while True:
            i = 10

    def pipe(max_batch_size, input_data, device):
        if False:
            i = 10
            return i + 15
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.coord_flip(data)
        pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, custom_shape_generator(150, 250, 2, 2)), pipe)

def test_lookup_table():
    if False:
        for i in range(10):
            print('nop')

    def pipe(max_batch_size, input_data, device):
        if False:
            i = 10
            return i + 15
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        processed = fn.lookup_table(data, keys=[1, 3], values=[10, 50])
        pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, array_1d_shape_generator, lo=0, hi=5, dtype=np.uint8), pipe)

def test_reduce():
    if False:
        while True:
            i = 10
    reduce_fns = [fn.reductions.std_dev, fn.reductions.variance]

    def pipe(max_batch_size, input_data, device, reduce_fn):
        if False:
            for i in range(10):
                print('nop')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        mean = fn.reductions.mean(data)
        reduced = reduce_fn(data, mean)
        pipe.set_outputs(reduced)
        return pipe
    for rf in reduce_fns:
        check_pipeline(generate_data(31, 13, image_like_shape_generator), pipe, reduce_fn=rf)

def test_sequence_rearrange():
    if False:
        while True:
            i = 10

    def pipe(max_batch_size, input_data, device):
        if False:
            while True:
                i = 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device, layout='FHWC')
        processed = fn.sequence_rearrange(data, new_order=[0, 4, 1, 3, 2])
        pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, (5, 10, 20, 3), lo=0, hi=255, dtype=np.uint8), pipe)

def test_element_extract():
    if False:
        i = 10
        return i + 15

    def pipe(max_batch_size, input_data, device):
        if False:
            return 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device, layout='FHWC')
        (processed, _) = fn.element_extract(data, element_map=[0, 3])
        pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, (5, 10, 20, 3), lo=0, hi=255, dtype=np.uint8), pipe)

def test_nonsilent_region():
    if False:
        while True:
            i = 10

    def pipe(max_batch_size, input_data, device):
        if False:
            i = 10
            return i + 15
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        (processed, _) = fn.nonsilent_region(data)
        pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, array_1d_shape_generator, lo=0, hi=255, dtype=np.uint8), pipe, devices=['cpu'])

def test_mel_filter_bank():
    if False:
        while True:
            i = 10

    def pipe(max_batch_size, input_data, device):
        if False:
            print('Hello World!')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            data = fn.external_source(source=input_data, cycle=False, device=device)
            spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
            processed = fn.mel_filter_bank(spectrum)
            pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, array_1d_shape_generator), pipe)

def test_mfcc():
    if False:
        print('Hello World!')

    def pipe(max_batch_size, input_data, device):
        if False:
            i = 10
            return i + 15
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device)
        spectrum = fn.spectrogram(data, nfft=60, window_length=50, window_step=25)
        mel = fn.mel_filter_bank(spectrum)
        dec = fn.to_decibels(mel)
        processed = fn.mfcc(dec)
        pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, array_1d_shape_generator), pipe)

@nottest
def generate_decoders_data(data_dir, data_extension, exclude_subdirs=[]):
    if False:
        i = 10
        return i + 15
    fnames = test_utils.filter_files(data_dir, data_extension, exclude_subdirs=exclude_subdirs)
    nfiles = len(fnames)
    for i in range(len(fnames), 10):
        fnames.append(fnames[-1])
    nfiles = len(fnames)
    _input_epoch = [list(map(lambda fname: test_utils.read_file_bin(fname), fnames[:nfiles // 3])), list(map(lambda fname: test_utils.read_file_bin(fname), fnames[nfiles // 3:nfiles // 2])), list(map(lambda fname: test_utils.read_file_bin(fname), fnames[nfiles // 2:]))]
    input_epoch = []
    for inp in _input_epoch:
        max_len = max((sample.shape[0] for sample in inp))
        inp = map(lambda sample: np.pad(sample, (0, max_len - sample.shape[0])), inp)
        input_epoch.append(np.stack(list(inp)))
    input_epoch = list(map(lambda batch: np.reshape(batch, batch.shape), input_epoch))
    return input_epoch

@nottest
def test_decoders_check(pipeline_fn, data_dir, data_extension, devices=['cpu'], exclude_subdirs=[]):
    if False:
        return 10
    data = generate_decoders_data(data_dir=data_dir, data_extension=data_extension, exclude_subdirs=exclude_subdirs)
    check_pipeline(data, pipeline_fn=pipeline_fn, devices=devices)

@nottest
def test_decoders_run(pipeline_fn, data_dir, data_extension, devices=['cpu'], exclude_subdirs=[]):
    if False:
        while True:
            i = 10
    data = generate_decoders_data(data_dir=data_dir, data_extension=data_extension, exclude_subdirs=exclude_subdirs)
    run_pipeline(data, pipeline_fn=pipeline_fn, devices=devices)

def test_audio_decoders():
    if False:
        print('Hello World!')

    def audio_decoder_pipe(max_batch_size, input_data, device):
        if False:
            return 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device='cpu')
        (decoded, _) = fn.decoders.audio(encoded, downmix=True, sample_rate=12345, device=device)
        pipe.set_outputs(decoded)
        return pipe
    audio_path = os.path.join(test_utils.get_dali_extra_path(), 'db', 'audio')
    yield (test_decoders_check, audio_decoder_pipe, audio_path, '.wav')

def test_image_decoders():
    if False:
        for i in range(10):
            print('nop')

    def image_decoder_pipe(module, max_batch_size, input_data, device):
        if False:
            for i in range(10):
                print('nop')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device='cpu')
        decoded = module.image(encoded, device=device)
        pipe.set_outputs(decoded)
        return pipe

    def image_decoder_crop_pipe(module, max_batch_size, input_data, device):
        if False:
            return 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device='cpu')
        decoded = module.image_crop(encoded, device=device)
        pipe.set_outputs(decoded)
        return pipe

    def image_decoder_slice_pipe(module, max_batch_size, input_data, device):
        if False:
            for i in range(10):
                print('nop')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device='cpu')
        decoded = module.image_slice(encoded, 0.1, 0.4, axes=0, device=device)
        pipe.set_outputs(decoded)
        return pipe

    def image_decoder_rcrop_pipe(module, max_batch_size, input_data, device):
        if False:
            i = 10
            return i + 15
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device='cpu')
        decoded = module.image_random_crop(encoded, device=device)
        pipe.set_outputs(decoded)
        return pipe

    def peek_image_shape_pipe(module, max_batch_size, input_data, device):
        if False:
            print('Hello World!')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device='cpu')
        shape = module.peek_image_shape(encoded, device=device)
        pipe.set_outputs(shape)
        return pipe
    image_decoder_extensions = ['.jpg', '.bmp', '.png', '.pnm', '.jp2']
    image_decoder_pipes = [image_decoder_pipe, image_decoder_crop_pipe, image_decoder_slice_pipe]
    data_path = os.path.join(test_utils.get_dali_extra_path(), 'db', 'single')
    exclude_subdirs = ['jpeg_lossless']
    for ext in image_decoder_extensions:
        for pipe_template in image_decoder_pipes:
            pipe = partial(pipe_template, fn.decoders)
            yield (test_decoders_check, pipe, data_path, ext, ['cpu', 'mixed'], exclude_subdirs)
            pipe = partial(pipe_template, fn.experimental.decoders)
            yield (test_decoders_check, pipe, data_path, ext, ['cpu', 'mixed'], exclude_subdirs)
        pipe = partial(image_decoder_rcrop_pipe, fn.decoders)
        yield (test_decoders_run, pipe, data_path, ext, ['cpu', 'mixed'], exclude_subdirs)
        pipe = partial(image_decoder_rcrop_pipe, fn.experimental.decoders)
        yield (test_decoders_run, pipe, data_path, ext, ['cpu', 'mixed'], exclude_subdirs)
    pipe = partial(peek_image_shape_pipe, fn)
    yield (test_decoders_check, pipe, data_path, '.jpg', ['cpu'], exclude_subdirs)
    pipe = partial(peek_image_shape_pipe, fn.experimental)
    yield (test_decoders_check, pipe, data_path, '.jpg', ['cpu'], exclude_subdirs)

def test_python_function():
    if False:
        return 10

    def resize(data):
        if False:
            return 10
        data += 13
        return data

    def pipe(max_batch_size, input_data, device):
        if False:
            for i in range(10):
                print('nop')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0, exec_async=False, exec_pipelined=False)
        with pipe:
            data = fn.external_source(source=input_data, cycle=False, device=device)
            processed = fn.python_function(data, function=resize, num_outputs=1)
            pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, image_like_shape_generator), pipe, devices=['cpu'])

def test_reinterpret():
    if False:
        return 10

    def pipe(max_batch_size, input_data, device, input_layout):
        if False:
            for i in range(10):
                print('nop')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device, layout=input_layout)
        processed = fn.reinterpret(data, rel_shape=[0.5, 1, -1])
        pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, (160, 80, 3), lo=0, hi=255, dtype=np.uint8), pipeline_fn=pipe, input_layout='HWC')
    check_pipeline(generate_data(31, 13, (5, 160, 80, 3), lo=0, hi=255, dtype=np.uint8), pipeline_fn=pipe, input_layout='FHWC')

def test_segmentation_select_masks():
    if False:
        for i in range(10):
            print('nop')

    def get_data_source(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return make_batch_select_masks(*args, **kwargs)

    def pipe(max_batch_size, input_data, device):
        if False:
            print('Hello World!')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=None, seed=1234)
        with pipe:
            (polygons, vertices, selected_masks) = fn.external_source(num_outputs=3, device=device, source=input_data)
            (out_polygons, out_vertices) = fn.segmentation.select_masks(selected_masks, polygons, vertices, reindex_masks=False)
        pipe.set_outputs(polygons, vertices, selected_masks, out_polygons, out_vertices)
        return pipe
    input_data = [get_data_source(random.randint(5, 31), vertex_ndim=2, npolygons_range=(1, 5), nvertices_range=(3, 10)) for _ in range(13)]
    check_pipeline(input_data, pipeline_fn=pipe, devices=['cpu'])

def test_optical_flow():
    if False:
        return 10
    if not is_of_supported():
        raise nose.SkipTest('Optical Flow is not supported on this platform')

    def pipe(max_batch_size, input_data, device, input_layout=None):
        if False:
            print('Hello World!')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        with pipe:
            data = fn.external_source(device=device, source=input_data, cycle=False, layout=input_layout)
            processed = fn.optical_flow(data, device=device, output_grid=4)
        pipe.set_outputs(processed)
        return pipe
    max_batch_size = 5
    bach_sizes = [max_batch_size // 2, max_batch_size // 4, max_batch_size]
    input_data = [[load_frames() for _ in range(bs)] for bs in bach_sizes]
    check_pipeline(input_data, pipeline_fn=pipe, devices=['gpu'], input_layout='FHWC')

def test_tensor_subscript():
    if False:
        for i in range(10):
            print('nop')

    def pipe(max_batch_size, input_data, device, input_layout):
        if False:
            return 10
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        data = fn.external_source(source=input_data, cycle=False, device=device, layout=input_layout)
        processed = data[2, :-2, 1]
        pipe.set_outputs(processed)
        return pipe
    check_pipeline(generate_data(31, 13, (160, 80, 3), lo=0, hi=255, dtype=np.uint8), pipeline_fn=pipe, input_layout='HWC')
    check_pipeline(generate_data(31, 13, (5, 160, 80, 3), lo=0, hi=255, dtype=np.uint8), pipeline_fn=pipe, input_layout='FHWC')

def test_subscript_dim_check():
    if False:
        i = 10
        return i + 15
    data = generate_data(31, 13, array_1d_shape_generator, lo=0, hi=255, dtype=np.uint8)
    check_pipeline(data, single_op_pipeline, operator_fn=fn.subscript_dim_check, num_subscripts=1)

def test_crop_argument_from_external_source():
    if False:
        while True:
            i = 10
    '\n    Tests, if the fn.crop operator works correctly, when its actual batch size is lower\n    than max batch size.\n    '

    @pipeline_def(batch_size=32, num_threads=4, device_id=0)
    def pipeline():
        if False:
            while True:
                i = 10
        images = fn.external_source(device='cpu', name='IMAGE', no_copy=False)
        crop_x = fn.external_source(device='cpu', name='CROP_X', no_copy=False)
        images = fn.decoders.image(images, device='mixed')
        images = fn.crop(images, crop_pos_x=crop_x, crop_pos_y=0.05, crop_w=113, crop_h=149)
        return images
    pipe = pipeline()
    pipe.build()
    image_data = np.fromfile(os.path.join(test_utils.get_dali_extra_path(), 'db', 'single', 'jpeg', '100', 'swan-3584559_640.jpg'), dtype=np.uint8)
    pipe.feed_input('IMAGE', [image_data])
    pipe.feed_input('CROP_X', [np.float32(0.5)])
    pipe.feed_input('IMAGE', [image_data])
    pipe.feed_input('CROP_X', [np.float32(0.4)])
    pipe.run()

def test_video_decoder():
    if False:
        for i in range(10):
            print('nop')

    def video_decoder_pipe(max_batch_size, input_data, device):
        if False:
            for i in range(10):
                print('nop')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        encoded = fn.external_source(source=input_data, cycle=False, device='cpu')
        decoded = fn.experimental.decoders.video(encoded, device=device)
        pipe.set_outputs(decoded)
        return pipe
    file_path = os.path.join(test_utils.get_dali_extra_path(), 'db', 'video', 'cfr', 'test_1.mp4')
    video_file = np.fromfile(file_path, dtype=np.uint8)
    batches = [[video_file] * 2, [video_file] * 5, [video_file] * 3]
    check_pipeline(batches, video_decoder_pipe, devices=['cpu', 'mixed'])

@has_operator('experimental.inflate')
@restrict_platform(min_compute_cap=6.0, platforms=['x86_64'])
def test_inflate():
    if False:
        for i in range(10):
            print('nop')
    import lz4.block

    def sample_to_lz4(sample):
        if False:
            while True:
                i = 10
        deflated_buf = lz4.block.compress(sample, store_size=False)
        return np.frombuffer(deflated_buf, dtype=np.uint8)

    def inflate_pipline(max_batch_size, inputs, device):
        if False:
            for i in range(10):
                print('nop')
        input_data = [[sample_to_lz4(sample) for sample in batch] for batch in inputs]
        input_shape = [[np.array(sample.shape, dtype=np.int32) for sample in batch] for batch in inputs]

        @pipeline_def
        def piepline():
            if False:
                while True:
                    i = 10
            defalted = fn.external_source(source=input_data)
            shape = fn.external_source(source=input_shape)
            return fn.experimental.inflate(defalted.gpu(), shape=shape)
        return piepline(batch_size=max_batch_size, num_threads=4, device_id=0)

    def sample_gen():
        if False:
            return 10
        j = 42
        while True:
            yield np.full((13, 7), j)
            j += 1
    sample = sample_gen()
    batches = [[next(sample) for _ in range(5)], [next(sample) for _ in range(13)], [next(sample) for _ in range(2)]]
    check_pipeline(batches, inflate_pipline, devices=['gpu'])

def test_debayer():
    if False:
        i = 10
        return i + 15
    from debayer_test_utils import rgb2bayer, bayer_patterns, blue_position

    def debayer_pipline(max_batch_size, inputs, device):
        if False:
            print('Hello World!')
        batches = [list(zip(*batch)) for batch in inputs]
        img_batches = [list(imgs) for (imgs, _) in batches]
        blue_positions = [list(positions) for (_, positions) in batches]

        @pipeline_def
        def piepline():
            if False:
                print('Hello World!')
            bayered = fn.external_source(source=img_batches)
            positions = fn.external_source(source=blue_positions)
            return fn.experimental.debayer(bayered.gpu(), blue_position=positions)
        return piepline(batch_size=max_batch_size, num_threads=4, device_id=0)

    def sample_gen():
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(seed=101)
        j = 0
        while True:
            pattern = bayer_patterns[j % len(bayer_patterns)]
            (h, w) = 2 * np.int32(rng.uniform(2, 3, 2))
            (r, g, b) = (np.full((h, w), j), np.full((h, w), j + 1), np.full((h, w), j + 2))
            rgb = np.uint8(np.stack([r, g, b], axis=2))
            yield (rgb2bayer(rgb, pattern), np.array(blue_position(pattern), dtype=np.int32))
            j += 1
    sample = sample_gen()
    batches = [[next(sample) for _ in range(5)], [next(sample) for _ in range(13)], [next(sample) for _ in range(2)]]
    check_pipeline(batches, debayer_pipline, devices=['gpu'])

def test_filter():
    if False:
        print('Hello World!')

    def filter_pipeline(max_batch_size, inputs, device):
        if False:
            for i in range(10):
                print('nop')
        batches = [list(zip(*batch)) for batch in inputs]
        sample_batches = [list(inp_batch) for (inp_batch, _, _) in batches]
        filter_batches = [list(filt_batch) for (_, filt_batch, _) in batches]
        fill_value_bacthes = [list(fvs) for (_, _, fvs) in batches]

        @pipeline_def
        def pipeline():
            if False:
                i = 10
                return i + 15
            samples = fn.external_source(source=sample_batches, layout='HWC')
            filters = fn.external_source(source=filter_batches)
            fill_values = fn.external_source(source=fill_value_bacthes)
            return fn.experimental.filter(samples.gpu(), filters, fill_values, border='constant')
        return pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)

    def sample_gen():
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(seed=101)
        sample_shapes = [(300, 600, 3), (100, 100, 3), (500, 1024, 1), (40, 40, 20)]
        filter_shapes = [(5, 7), (3, 3), (60, 2)]
        j = 0
        while True:
            sample_shape = sample_shapes[j % len(sample_shapes)]
            filter_shape = filter_shapes[j % len(filter_shapes)]
            sample = np.uint8(rng.uniform(0, 255, sample_shape))
            filter = np.float32(rng.uniform(0, 255, filter_shape))
            yield (sample, filter, np.array([rng.uniform(0, 255)], dtype=np.uint8))
            j += 1
    sample = sample_gen()
    batches = [[next(sample) for _ in range(5)], [next(sample) for _ in range(13)], [next(sample) for _ in range(2)]]
    check_pipeline(batches, filter_pipeline, devices=['gpu'])

def test_cast_like():
    if False:
        i = 10
        return i + 15

    def pipe(max_batch_size, input_data, device):
        if False:
            for i in range(10):
                print('nop')
        pipe = Pipeline(batch_size=max_batch_size, num_threads=4, device_id=0)
        (data, data2) = fn.external_source(source=input_data, cycle=False, device=device, num_outputs=2)
        out = fn.cast_like(data, data2)
        pipe.set_outputs(out)
        return pipe

    def get_data(batch_size):
        if False:
            return 10
        test_data_shape = [random.randint(5, 21), random.randint(5, 21), 3]
        data1 = [np.random.randint(0, 255, size=test_data_shape, dtype=np.uint8) for _ in range(batch_size)]
        data2 = [np.random.randint(1, 4, size=test_data_shape, dtype=np.int32) for _ in range(batch_size)]
        return (data1, data2)
    input_data = [get_data(random.randint(5, 31)) for _ in range(13)]
    check_pipeline(input_data, pipeline_fn=pipe)

def test_conditional():
    if False:
        i = 10
        return i + 15

    def conditional_wrapper(max_batch_size, input_data, device):
        if False:
            while True:
                i = 10

        @experimental_pipeline_def(enable_conditionals=True, batch_size=max_batch_size, num_threads=4, device_id=0)
        def actual_pipe():
            if False:
                return 10
            variable_condition = fn.external_source(source=input_data, cycle=False, device=device)
            variable_data = variable_condition + 42.0
            if variable_condition:
                other_variable_data = variable_condition + 100
                output = variable_data + other_variable_data
            else:
                output = types.Constant(np.array(42.0), device='cpu')
            logical_expr = variable_condition or not variable_condition
            logical_expr2 = not variable_condition and variable_condition
            return (output, variable_condition, variable_data, logical_expr, logical_expr2)
        return actual_pipe()
    check_pipeline(generate_data(31, 13, custom_shape_generator(), lo=False, hi=True, dtype=np.bool_), pipeline_fn=conditional_wrapper, devices=['cpu'])

    def split_merge_wrapper(max_batch_size, input_data, device):
        if False:
            while True:
                i = 10

        @experimental_pipeline_def(enable_conditionals=True, batch_size=max_batch_size, num_threads=4, device_id=0)
        def actual_pipe():
            if False:
                return 10
            variable_pred = fn.external_source(source=input_data, cycle=False, device=device)
            variable_data = variable_pred + 42.0
            (true, false) = fn._conditional.split(variable_data, predicate=variable_pred)
            true = true + 10.0
            merged = fn._conditional.merge(true, false, predicate=variable_pred)
            return (merged, variable_pred)
        return actual_pipe()
    check_pipeline(generate_data(31, 13, custom_shape_generator(), lo=False, hi=True, dtype=np.bool_), pipeline_fn=split_merge_wrapper, devices=['cpu'])

    def not_validate_wrapper(max_batch_size, input_data, device):
        if False:
            return 10

        @experimental_pipeline_def(enable_conditionals=True, batch_size=max_batch_size, num_threads=4, device_id=0)
        def actual_pipe():
            if False:
                return 10
            variable_pred = fn.external_source(source=input_data, cycle=False, device=device)
            negated = fn._conditional.not_(variable_pred)
            validated = fn._conditional.validate_logical(variable_pred, expression_name='or', expression_side='right')
            return (negated, validated, variable_pred)
        return actual_pipe()
    check_pipeline(generate_data(31, 13, custom_shape_generator(), lo=False, hi=True, dtype=np.bool_), pipeline_fn=not_validate_wrapper, devices=['cpu'])
tested_methods = ['_conditional.merge', '_conditional.split', '_conditional.not_', '_conditional.validate_logical', 'arithmetic_generic_op', 'audio_decoder', 'audio_resample', 'batch_permutation', 'bb_flip', 'bbox_paste', 'box_encoder', 'brightness', 'brightness_contrast', 'cast', 'cast_like', 'cat', 'coin_flip', 'color_space_conversion', 'color_twist', 'constant', 'contrast', 'coord_flip', 'coord_transform', 'copy', 'crop', 'crop_mirror_normalize', 'decoders.audio', 'decoders.image', 'decoders.image_crop', 'decoders.image_random_crop', 'decoders.image_slice', 'dl_tensor_python_function', 'dump_image', 'experimental.equalize', 'element_extract', 'erase', 'expand_dims', 'experimental.debayer', 'experimental.decoders.image', 'experimental.decoders.image_crop', 'experimental.decoders.image_slice', 'experimental.decoders.image_random_crop', 'experimental.decoders.video', 'experimental.filter', 'experimental.inflate', 'experimental.median_blur', 'experimental.peek_image_shape', 'experimental.remap', 'external_source', 'fast_resize_crop_mirror', 'flip', 'gaussian_blur', 'get_property', 'grid_mask', 'hsv', 'hue', 'image_decoder', 'image_decoder_crop', 'image_decoder_random_crop', 'image_decoder_slice', 'jitter', 'jpeg_compression_distortion', 'laplacian', 'lookup_table', 'math.abs', 'math.acos', 'math.acosh', 'math.asin', 'math.asinh', 'math.atan', 'math.atan2', 'math.atanh', 'math.cbrt', 'math.ceil', 'math.clamp', 'math.cos', 'math.cosh', 'math.exp', 'math.fabs', 'math.floor', 'math.fpow', 'math.log', 'math.log10', 'math.log2', 'math.max', 'math.min', 'math.pow', 'math.rsqrt', 'math.sin', 'math.sinh', 'math.sqrt', 'math.tan', 'math.tanh', 'mel_filter_bank', 'mfcc', 'noise.gaussian', 'noise.salt_and_pepper', 'noise.shot', 'nonsilent_region', 'normal_distribution', 'normalize', 'numba.fn.experimental.numba_function', 'one_hot', 'optical_flow', 'pad', 'paste', 'peek_image_shape', 'per_frame', 'permute_batch', 'power_spectrum', 'preemphasis_filter', 'python_function', 'random.coin_flip', 'random.normal', 'random.uniform', 'random_bbox_crop', 'random_resized_crop', 'reductions.max', 'reductions.mean', 'reductions.mean_square', 'reductions.min', 'reductions.rms', 'reductions.std_dev', 'reductions.sum', 'reductions.variance', 'reinterpret', 'reshape', 'resize', 'resize_crop_mirror', 'experimental.tensor_resize', 'roi_random_crop', 'rotate', 'saturation', 'segmentation.random_mask_pixel', 'segmentation.random_object_bbox', 'segmentation.select_masks', 'sequence_rearrange', 'shapes', 'slice', 'spectrogram', 'sphere', 'squeeze', 'ssd_random_crop', 'stack', 'subscript_dim_check', 'tensor_subscript', 'to_decibels', 'transform_translation', 'transforms.combine', 'transforms.crop', 'transforms.rotation', 'transforms.scale', 'transforms.shear', 'transforms.translation', 'transpose', 'uniform', 'warp_affine', 'water']
excluded_methods = ['hidden.*', '_conditional.hidden.*', 'multi_paste', 'coco_reader', 'sequence_reader', 'numpy_reader', 'file_reader', 'caffe_reader', 'caffe2_reader', 'mxnet_reader', 'tfrecord_reader', 'nemo_asr_reader', 'video_reader', 'video_reader_resize', 'readers.coco', 'readers.sequence', 'readers.numpy', 'readers.file', 'readers.caffe', 'readers.caffe2', 'readers.mxnet', 'readers.tfrecord', 'readers.nemo_asr', 'readers.video', 'readers.video_resize', 'readers.webdataset', 'experimental.inputs.video', 'experimental.readers.video', 'experimental.audio_resample', 'experimental.readers.fits']

def test_coverage():
    if False:
        return 10
    methods = module_functions(fn, remove_prefix='nvidia.dali.fn', allowed_private_modules=['_conditional'])
    methods += module_functions(dmath, remove_prefix='nvidia.dali')
    exclude = '|'.join(['(^' + x.replace('.', '\\.').replace('*', '.*').replace('?', '.') + '$)' for x in excluded_methods])
    exclude = re.compile(exclude)
    methods = [x for x in methods if not exclude.match(x)]
    assert set(methods).difference(set(tested_methods)) == set(), "Test doesn't cover:\n {}".format(set(methods) - set(tested_methods))