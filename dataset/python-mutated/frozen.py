__all__ = ['current_model', 'model_context']
import copy
from typing import Optional
from nni.mutable import frozen_context, Sample

def current_model() -> Optional[Sample]:
    if False:
        return 10
    'Get the current model sample in :func:`model_context`.\n\n    The sample is supposed to be the same as :attr:`nni.nas.space.ExecutableModelSpace.sample`.\n\n    This method is only valid when called inside :func:`model_context`.\n    By default, only the execution of :class:`~nni.nas.space.SimplifiedModelSpace` will set the context,\n    so that :func:`current_model` is meaningful within the re-instantiation of the model.\n\n    Returns\n    -------\n    Model sample (i.e., architecture dict) before freezing, produced by strategy.\n    If not called inside :func:`model_context`, returns None.\n    '
    cur = frozen_context.current()
    if cur is None or not cur.get('__arch__'):
        return None
    cur = copy.copy(cur)
    cur.pop('__arch__')
    return cur

def model_context(sample: Sample) -> frozen_context:
    if False:
        print('Hello World!')
    'Get a context stack of the current model sample (i.e., architecture dict).\n\n    This should be used together with :func:`current_model`.\n\n    :func:`model_context` is read-only, and should not be used to modify the architecture dict.\n    '
    return frozen_context({**sample, '__arch__': True})