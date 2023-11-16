"""Pickler for values, functions, and classes.

For internal use only. No backwards compatibility guarantees.

Pickles created by the pickling library contain non-ASCII characters, so
we base64-encode the results so that we can put them in a JSON objects.
The pickler is used to embed FlatMap callable objects into the workflow JSON
description.

The pickler module should be used to pickle functions and modules; for values,
the coders.*PickleCoder classes should be used instead.
"""
from apache_beam.internal import cloudpickle_pickler
from apache_beam.internal import dill_pickler
USE_CLOUDPICKLE = 'cloudpickle'
USE_DILL = 'dill'
DEFAULT_PICKLE_LIB = USE_DILL
desired_pickle_lib = dill_pickler

def dumps(o, enable_trace=True, use_zlib=False):
    if False:
        return 10
    return desired_pickle_lib.dumps(o, enable_trace=enable_trace, use_zlib=use_zlib)

def loads(encoded, enable_trace=True, use_zlib=False):
    if False:
        i = 10
        return i + 15
    'For internal use only; no backwards-compatibility guarantees.'
    return desired_pickle_lib.loads(encoded, enable_trace=enable_trace, use_zlib=use_zlib)

def dump_session(file_path):
    if False:
        print('Hello World!')
    'For internal use only; no backwards-compatibility guarantees.\n\n  Pickle the current python session to be used in the worker.\n  '
    return desired_pickle_lib.dump_session(file_path)

def load_session(file_path):
    if False:
        while True:
            i = 10
    return desired_pickle_lib.load_session(file_path)

def set_library(selected_library=DEFAULT_PICKLE_LIB):
    if False:
        for i in range(10):
            print('nop')
    ' Sets pickle library that will be used. '
    global desired_pickle_lib
    if (selected_library == USE_DILL) != (desired_pickle_lib == dill_pickler):
        dill_pickler.override_pickler_hooks(selected_library == USE_DILL)
    if selected_library == 'default':
        selected_library = DEFAULT_PICKLE_LIB
    if selected_library == USE_DILL:
        desired_pickle_lib = dill_pickler
    elif selected_library == USE_CLOUDPICKLE:
        desired_pickle_lib = cloudpickle_pickler
    else:
        raise ValueError(f'Unknown pickler library: {selected_library}')