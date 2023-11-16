import os
from cupy.testing._pytest_impl import is_available, check_available
if is_available():
    import pytest
    _gpu_limit = int(os.getenv('CUPY_TEST_GPU_LIMIT', '-1'))

    def slow(*args, **kwargs):
        if False:
            while True:
                i = 10
        return pytest.mark.slow(*args, **kwargs)
else:

    def _dummy_callable(*args, **kwargs):
        if False:
            print('Hello World!')
        check_available('pytest attributes')
        assert False
    slow = _dummy_callable

def multi_gpu(gpu_num):
    if False:
        print('Hello World!')
    'Decorator to indicate number of GPUs required to run the test.\n\n    Tests can be annotated with this decorator (e.g., ``@multi_gpu(2)``) to\n    declare number of GPUs required to run. When running tests, if\n    ``CUPY_TEST_GPU_LIMIT`` environment variable is set to value greater\n    than or equals to 0, test cases that require GPUs more than the limit will\n    be skipped.\n    '
    check_available('multi_gpu attribute')
    assert 1 < gpu_num

    def _wrapper(f):
        if False:
            i = 10
            return i + 15
        return pytest.mark.skipif(0 <= _gpu_limit < gpu_num, reason='{} GPUs required'.format(gpu_num))(pytest.mark.multi_gpu(f))
    return _wrapper