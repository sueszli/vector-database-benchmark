from nvidia.dali import Pipeline, pipeline_def
from nose.tools import nottest
from nose_utils import raises
import nvidia.dali.fn as fn
from test_utils import get_dali_extra_path, compare_pipelines
import os
data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, 'db', 'single', 'jpeg')
N_ITER = 2
max_batch_size = 4
num_threads = 4
device_id = 0

def reference_pipeline(flip_vertical, flip_horizontal, ref_batch_size=max_batch_size):
    if False:
        i = 10
        return i + 15
    pipeline = Pipeline(ref_batch_size, num_threads, device_id)
    with pipeline:
        (data, _) = fn.readers.file(file_root=images_dir)
        img = fn.decoders.image(data)
        flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
        pipeline.set_outputs(flipped, img)
    return pipeline

@nottest
@pipeline_def(batch_size=max_batch_size, num_threads=num_threads, device_id=device_id)
def pipeline_static(flip_vertical, flip_horizontal):
    if False:
        print('Hello World!')
    (data, _) = fn.readers.file(file_root=images_dir)
    img = fn.decoders.image(data)
    flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
    return (flipped, img)

@nottest
@pipeline_def
def pipeline_runtime(flip_vertical, flip_horizontal):
    if False:
        print('Hello World!')
    (data, _) = fn.readers.file(file_root=images_dir)
    img = fn.decoders.image(data)
    flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
    return (flipped, img)

@nottest
def test_pipeline_static(flip_vertical, flip_horizontal):
    if False:
        for i in range(10):
            print('nop')
    put_args = pipeline_static(flip_vertical, flip_horizontal)
    ref = reference_pipeline(flip_vertical, flip_horizontal)
    compare_pipelines(put_args, ref, batch_size=max_batch_size, N_iterations=N_ITER)

@nottest
def test_pipeline_runtime(flip_vertical, flip_horizontal):
    if False:
        for i in range(10):
            print('nop')
    put_combined = pipeline_runtime(flip_vertical, flip_horizontal, batch_size=max_batch_size, num_threads=num_threads, device_id=device_id)
    ref = reference_pipeline(flip_vertical, flip_horizontal)
    compare_pipelines(put_combined, ref, batch_size=max_batch_size, N_iterations=N_ITER)

@nottest
def test_pipeline_override(flip_vertical, flip_horizontal, batch_size):
    if False:
        print('Hello World!')
    put_combined = pipeline_static(flip_vertical, flip_horizontal, batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    ref = reference_pipeline(flip_vertical, flip_horizontal, ref_batch_size=batch_size)
    compare_pipelines(put_combined, ref, batch_size=batch_size, N_iterations=N_ITER)

def test_pipeline_decorator():
    if False:
        print('Hello World!')
    for vert in [0, 1]:
        for hori in [0, 1]:
            yield (test_pipeline_static, vert, hori)
            yield (test_pipeline_runtime, vert, hori)
            yield (test_pipeline_override, vert, hori, 5)
    yield (test_pipeline_runtime, fn.random.coin_flip(seed=123), fn.random.coin_flip(seed=234))
    yield (test_pipeline_static, fn.random.coin_flip(seed=123), fn.random.coin_flip(seed=234))

def test_duplicated_argument():
    if False:
        for i in range(10):
            print('nop')

    @pipeline_def(batch_size=max_batch_size, num_threads=num_threads, device_id=device_id)
    def ref_pipeline(val):
        if False:
            for i in range(10):
                print('nop')
        (data, _) = fn.readers.file(file_root=images_dir)
        return data + val

    @pipeline_def(batch_size=max_batch_size, num_threads=num_threads, device_id=device_id)
    def pipeline_duplicated_arg(max_streams):
        if False:
            return 10
        (data, _) = fn.readers.file(file_root=images_dir)
        return data + max_streams
    pipe = pipeline_duplicated_arg(max_streams=42)
    assert pipe._max_streams == -1
    ref = ref_pipeline(42)
    compare_pipelines(pipe, ref, batch_size=max_batch_size, N_iterations=N_ITER)

@pipeline_def
def pipeline_kwargs(arg1, arg2, *args, **kwargs):
    if False:
        print('Hello World!')
    pass

@raises(TypeError, regex='\\*\\*kwargs.*not allowed')
def test_kwargs_exception():
    if False:
        return 10
    pipeline_kwargs(arg1=1, arg2=2, arg3=3)

def test_is_pipeline_def():
    if False:
        i = 10
        return i + 15

    @pipeline_def
    def pipe():
        if False:
            for i in range(10):
                print('nop')
        return 42

    @pipeline_def()
    def pipe_unconf():
        if False:
            return 10
        return 42

    @pipeline_def(max_batch_size=1, num_threads=1, device_id=0)
    def pipe_conf():
        if False:
            for i in range(10):
                print('nop')
        return 42
    assert getattr(pipe, '_is_pipeline_def', False)
    assert getattr(pipe_unconf, '_is_pipeline_def', False)
    assert getattr(pipe_conf, '_is_pipeline_def', False)