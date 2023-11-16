import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def, Pipeline
import numpy as np
from nose_utils import raises

@pipeline_def(batch_size=8, num_threads=3, device_id=0)
def pipeline():
    if False:
        while True:
            i = 10
    output = fn.external_source(source=np.zeros((8, 8)), name='input')
    return output

@raises(RuntimeError, glob="Cannot use `feed_input` on the external source 'input' with a `source` argument specified.")
def test_feed_input_with_source():
    if False:
        return 10
    pipe = pipeline()
    pipe.build()
    pipe.feed_input('input', np.zeros((8, 8)))
    pipe.run()

def test_external_source_with_callback():
    if False:
        return 10
    "Test if using external_source with 'source' doesn't raise exceptions."
    pipe = pipeline()
    pipe.build()
    pipe.run()

def test_external_source_with_serialized_pipe():
    if False:
        return 10

    @pipeline_def
    def serialized_pipe():
        if False:
            i = 10
            return i + 15
        return fn.external_source(name='es')
    pipe = serialized_pipe(batch_size=10, num_threads=3, device_id=0)
    serialized_str = pipe.serialize()
    deserialized_pipe = Pipeline(10, 4, 0)
    deserialized_pipe.deserialize_and_build(serialized_str)
    deserialized_pipe.feed_input('es', np.zeros([10, 10]))