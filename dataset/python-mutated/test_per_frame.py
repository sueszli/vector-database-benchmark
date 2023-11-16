import numpy as np
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from test_utils import check_batch
from nose_utils import raises
batch_sizes = [5, 256, 128, 7]
max_batch_size = max(batch_sizes)

def input_batch(num_dim):
    if False:
        print('Hello World!')
    rng = np.random.default_rng(42)
    for batch_size in batch_sizes:
        yield [rng.random(rng.integers(low=0, high=50, size=num_dim)) for _ in range(batch_size)]

def run_pipeline(device, num_dim, replace=False, layout=None):
    if False:
        i = 10
        return i + 15

    @pipeline_def
    def pipeline():
        if False:
            while True:
                i = 10
        arg = fn.external_source(input_batch(num_dim), layout=layout)
        if device == 'gpu':
            arg = arg.gpu()
        return fn.per_frame(arg, replace=replace, device=device)
    pipe = pipeline(num_threads=4, batch_size=max_batch_size, device_id=0)
    pipe.build()
    expected_layout = 'F' + '*' * (num_dim - 1) if layout is None else 'F' + layout[1:]
    for baseline in input_batch(num_dim):
        (out,) = pipe.run()
        check_batch(out, baseline, len(baseline), expected_layout=expected_layout)

def test_set_layout():
    if False:
        for i in range(10):
            print('nop')
    for device in ['cpu', 'gpu']:
        for num_dim in (1, 2, 3):
            yield (run_pipeline, device, num_dim)

def test_replace_layout():
    if False:
        while True:
            i = 10
    for device in ['cpu', 'gpu']:
        for num_dim in (1, 2, 3):
            yield (run_pipeline, device, num_dim, True, 'XYZ'[:num_dim])

def test_verify_layout():
    if False:
        print('Hello World!')
    for device in ['cpu', 'gpu']:
        for num_dim in (1, 2, 3):
            yield (run_pipeline, device, num_dim, False, 'FYZ'[:num_dim])

def test_zero_dim_not_allowed():
    if False:
        for i in range(10):
            print('nop')
    expected_msg = 'Cannot mark zero-dimensional input as a sequence'
    for device in ['cpu', 'gpu']:
        yield (raises(RuntimeError, expected_msg)(run_pipeline), device, 0)

@raises(RuntimeError, "Per-frame argument input must be a sequence. The input layout should start with 'F'")
def _test_not_a_sequence_layout(device, num_dim, layout):
    if False:
        return 10
    run_pipeline(device, num_dim=num_dim, layout=layout)

def test_not_a_sequence_layout():
    if False:
        print('Hello World!')
    for device in ['cpu', 'gpu']:
        for num_dim in (1, 2, 3):
            yield (_test_not_a_sequence_layout, device, num_dim, 'XYZ'[:num_dim])

def _test_pass_through():
    if False:
        return 10

    @pipeline_def
    def pipeline():
        if False:
            i = 10
            return i + 15
        rng = fn.external_source(lambda info: np.array([info.iteration, info.iteration + 1], dtype=np.float32), batch=False)
        return fn.per_frame(fn.random.uniform(range=rng, device='gpu', shape=(1, 1, 1), seed=42))
    pipe = pipeline(batch_size=1, num_threads=4, device_id=0)
    pipe.build()
    for i in range(5):
        (out,) = pipe.run()
        [sample] = [np.array(s) for s in out.as_cpu()]
        assert i <= sample[0] < i + 1

def test_pass_through():
    if False:
        return 10
    for _ in range(50):
        _test_pass_through()