import pytest
from ray.air.execution.resources.request import ResourceRequest

def test_request_same():
    if False:
        print('Hello World!')
    'Test that resource requests are the same if they share the same properties.'
    assert ResourceRequest([{'CPU': 1}]) == ResourceRequest([{'CPU': 1}])
    assert ResourceRequest([{'CPU': 1}, {'CPU': 2}]) == ResourceRequest([{'CPU': 1}, {'CPU': 2}])
    assert ResourceRequest([{'CPU': 1, 'GPU': 1}]) == ResourceRequest([{'CPU': 1, 'GPU': 1}])
    assert ResourceRequest([{'CPU': 0, 'GPU': 1}]) == ResourceRequest([{'GPU': 1}])
    assert ResourceRequest([{'CPU': 1}], strategy='PACK') == ResourceRequest([{'CPU': 1}])
    assert ResourceRequest([{'CPU': 1}], strategy='PACK') != ResourceRequest([{'CPU': 1}], strategy='SPREAD')
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))