from collections import namedtuple
from metaflow.plugins.aws.aws_utils import compute_resource_attributes
MockDeco = namedtuple('MockDeco', ['name', 'attributes'])

def test_compute_resource_attributes():
    if False:
        i = 10
        return i + 15
    assert compute_resource_attributes([], MockDeco('batch', {}), {'cpu': '1'}) == {'cpu': '1'}
    assert compute_resource_attributes([], MockDeco('batch', {'cpu': 1}), {'cpu': '2'}) == {'cpu': '1'}
    assert compute_resource_attributes([], MockDeco('batch', {'cpu': '1'}), {'cpu': '2'}) == {'cpu': '1'}
    assert compute_resource_attributes([], MockDeco('batch', {'cpu': '1'}), {'cpu': '2', 'memory': '100'}) == {'cpu': '1', 'memory': '100'}
    assert compute_resource_attributes([], MockDeco('resources', {'cpu': '1'}), {'cpu': '2', 'memory': '100'}) == {'cpu': '1', 'memory': '100'}
    assert compute_resource_attributes([MockDeco('resources', {'cpu': '2'})], MockDeco('batch', {'cpu': 1}), {'cpu': '3'}) == {'cpu': '2.0'}
    assert compute_resource_attributes([MockDeco('resources', {'cpu': 0.83})], MockDeco('batch', {'cpu': '0.5'}), {'cpu': '1'}) == {'cpu': '0.83'}

def test_compute_resource_attributes_string():
    if False:
        return 10
    'Test string-valued resource attributes'
    assert compute_resource_attributes([], MockDeco('batch', {}), {'cpu': '1', 'instance_type': None}) == {'cpu': '1'}
    assert compute_resource_attributes([], MockDeco('batch', {'instance_type': 'p3.xlarge'}), {'cpu': '1', 'instance_type': None}) == {'cpu': '1', 'instance_type': 'p3.xlarge'}
    assert compute_resource_attributes([], MockDeco('batch', {'instance_type': 'p3.xlarge'}), {'cpu': '1', 'instance_type': 'p4.xlarge'}) == {'cpu': '1', 'instance_type': 'p3.xlarge'}
    assert compute_resource_attributes([], MockDeco('batch', {'instance_type': None}), {'cpu': '1', 'instance_type': 'p4.xlarge'}) == {'cpu': '1', 'instance_type': 'p4.xlarge'}