"""Helper library for functions used during TPU compilation."""
import contextlib
import threading

class TpuContext(threading.local):
    """A context object holding state about the TPU computation being built."""

    def __init__(self):
        if False:
            while True:
                i = 10
        'Creates a new TpuContext.'
        self._number_of_shards = None

    @property
    def number_of_shards(self):
        if False:
            while True:
                i = 10
        return self._number_of_shards

    def set_number_of_shards(self, number_of_shards):
        if False:
            for i in range(10):
                print('nop')
        self._number_of_shards = number_of_shards
_current_tpu_context = TpuContext()

@contextlib.contextmanager
def tpu_shard_context(number_of_shards):
    if False:
        for i in range(10):
            print('nop')
    'A context manager setting current number of shards.'
    if _current_tpu_context.number_of_shards is not None:
        raise NotImplementedError("tpu_shard_context cannot be nested.If you're using TPUEstimator with inference_on_tpu, make sure you have set export_saved_model_api_version=ExportSavedModelApiVersion.V2 in the creation of TPUEstimator.")
    try:
        _current_tpu_context.set_number_of_shards(number_of_shards)
        yield
    finally:
        _current_tpu_context.set_number_of_shards(None)

def get_tpu_context():
    if False:
        return 10
    return _current_tpu_context

def on_device_training_loop(func):
    if False:
        print('Hello World!')
    setattr(func, 'step_marker_location', 'STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP')
    return func