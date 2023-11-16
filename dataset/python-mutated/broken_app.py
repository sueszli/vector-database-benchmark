import ray.cloudpickle as pickle
from ray import serve

class NonserializableException(Exception):
    """This exception cannot be serialized."""

    def __reduce__(self):
        if False:
            return 10
        raise RuntimeError('This exception cannot be serialized!')
try:
    pickle.dumps(NonserializableException())
except RuntimeError as e:
    assert 'This exception cannot be serialized!' in repr(e)
raise NonserializableException('custom exception info')

@serve.deployment
def f():
    if False:
        for i in range(10):
            print('nop')
    pass
app = f.bind()