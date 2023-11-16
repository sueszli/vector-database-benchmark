"""
Test that fiftyone can be imported in a reasonable amount of time.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import warnings
import time
IMPORT_WARN_THRESHOLD = 3

def test_import_time(capsys):
    if False:
        return 10
    t1 = time.perf_counter()
    import fiftyone
    time_elapsed = time.perf_counter() - t1
    message = '`import fiftyone` took %f seconds' % time_elapsed
    if time_elapsed > IMPORT_WARN_THRESHOLD:
        warnings.warn(message)
        with capsys.disabled():
            print('\n::warning::%s\n' % message)