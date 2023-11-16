import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import test_utils
from nose_utils import raises

class TestPipeline(Pipeline):

    def __init__(self, batch_size, num_threads, shape):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(batch_size, num_threads, device_id=0, seed=42)
        self.cf = ops.random.Uniform(device='cpu', shape=shape, seed=42)

    def define_graph(self):
        if False:
            return 10
        cf = self.cf()
        return cf

def check_deserialization(batch_size, num_threads, shape):
    if False:
        print('Hello World!')
    ref_pipe = TestPipeline(batch_size=batch_size, num_threads=num_threads, shape=shape)
    serialized = ref_pipe.serialize()
    test_pipe = Pipeline.deserialize(serialized)
    test_utils.compare_pipelines(ref_pipe, test_pipe, batch_size=batch_size, N_iterations=3)

def check_deserialization_with_params(batch_size, num_threads, shape):
    if False:
        print('Hello World!')
    init_pipe = TestPipeline(batch_size=batch_size, num_threads=num_threads, shape=shape)
    serialized = init_pipe.serialize()
    ref_pipe = TestPipeline(batch_size=batch_size ** 2, num_threads=num_threads + 1, shape=shape)
    test_pipe = Pipeline.deserialize(serialized, batch_size=batch_size ** 2, num_threads=num_threads + 1)
    test_utils.compare_pipelines(ref_pipe, test_pipe, batch_size=batch_size ** 2, N_iterations=3)

def check_deserialization_from_file(batch_size, num_threads, shape):
    if False:
        while True:
            i = 10
    filename = '/tmp/dali.serialize.pipeline.test'
    ref_pipe = TestPipeline(batch_size=batch_size, num_threads=num_threads, shape=shape)
    ref_pipe.serialize(filename=filename)
    test_pipe = Pipeline.deserialize(filename=filename)
    test_utils.compare_pipelines(ref_pipe, test_pipe, batch_size=batch_size, N_iterations=3)

def check_deserialization_from_file_with_params(batch_size, num_threads, shape):
    if False:
        for i in range(10):
            print('nop')
    filename = '/tmp/dali.serialize.pipeline.test'
    init_pipe = TestPipeline(batch_size=batch_size, num_threads=num_threads, shape=shape)
    init_pipe.serialize(filename=filename)
    ref_pipe = TestPipeline(batch_size=batch_size ** 2, num_threads=num_threads + 1, shape=shape)
    test_pipe = Pipeline.deserialize(filename=filename, batch_size=batch_size ** 2, num_threads=num_threads + 1)
    test_utils.compare_pipelines(ref_pipe, test_pipe, batch_size=batch_size ** 2, N_iterations=3)

def test_deserialization():
    if False:
        return 10
    batch_sizes = [3]
    nums_thread = [1]
    shapes = [[6], [2, 5], [3, 1, 6]]
    for bs in batch_sizes:
        for nt in nums_thread:
            for sh in shapes:
                yield (check_deserialization, bs, nt, sh)

def test_deserialization_with_params():
    if False:
        for i in range(10):
            print('nop')
    batch_sizes = [3]
    nums_thread = [1]
    shapes = [[6], [2, 5], [3, 1, 6]]
    for bs in batch_sizes:
        for nt in nums_thread:
            for sh in shapes:
                yield (check_deserialization_with_params, bs, nt, sh)

def test_deserialization_from_file():
    if False:
        return 10
    batch_sizes = [3]
    nums_thread = [1]
    shapes = [[6], [2, 5], [3, 1, 6]]
    for bs in batch_sizes:
        for nt in nums_thread:
            for sh in shapes:
                yield (check_deserialization, bs, nt, sh)

def test_deserialization_from_file_with_params():
    if False:
        return 10
    batch_sizes = [3]
    nums_thread = [1]
    shapes = [[6], [2, 5], [3, 1, 6]]
    for bs in batch_sizes:
        for nt in nums_thread:
            for sh in shapes:
                yield (check_deserialization_with_params, bs, nt, sh)

@raises(ValueError, 'serialized_pipeline and filename arguments are mutually exclusive. Precisely one of them should be defined.')
def test_incorrect_invocation_mutually_exclusive_params():
    if False:
        for i in range(10):
            print('nop')
    filename = '/tmp/dali.serialize.pipeline.test'
    pipe = TestPipeline(batch_size=3, num_threads=1, shape=[666])
    serialized = pipe.serialize(filename=filename)
    Pipeline.deserialize(serialized_pipeline=serialized, filename=filename)

@raises(ValueError, 'serialized_pipeline and filename arguments are mutually exclusive. Precisely one of them should be defined.')
def test_incorrect_invocation_no_params():
    if False:
        for i in range(10):
            print('nop')
    Pipeline.deserialize()