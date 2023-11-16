from typing import List
import pytest
from ray.autoscaler._private.gcp.node_provider import _retry

class MockGCPNodeProvider:

    def __init__(self, errors: List[type]):
        if False:
            print('Hello World!')
        self.errors = errors
        self.error_index = -1
        self._construct_clients()

    def _construct_clients(self):
        if False:
            while True:
                i = 10
        self.error_index += 1

    @_retry
    def mock_method(self, *args, **kwargs):
        if False:
            print('Hello World!')
        error = self.errors[self.error_index]
        if error:
            raise error
        return (args, kwargs)
(B, V) = (BrokenPipeError, ValueError)

@pytest.mark.parametrize('error_input,expected_error_raised', [([None], None), ([B, B, B, B, None], None), ([B, B, V, B, None], V), ([B, B, B, B, B, None], B), ([B, B, B, B, B, B, None], B)])
def test_gcp_broken_pipe_retry(error_input, expected_error_raised):
    if False:
        i = 10
        return i + 15
    'Tests retries of BrokenPipeError in GCPNodeProvider.\n\n    Args:\n        error_input: List of exceptions hit during retries of test mock_method.\n            None means no exception.\n        expected_error_raised: Expected exception raised.\n            None means no exception.\n    '
    provider = MockGCPNodeProvider(error_input)
    if expected_error_raised:
        with pytest.raises(expected_error_raised):
            provider.mock_method(1, 2, a=4, b=5)
    else:
        ret = provider.mock_method(1, 2, a=4, b=5)
        assert ret == ((1, 2), {'a': 4, 'b': 5})
if __name__ == '__main__':
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))