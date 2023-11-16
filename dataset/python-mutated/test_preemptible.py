import uuid
import google.auth
import pytest
from ..instances.delete import delete_instance
from ..instances.preemptible.create_preemptible import create_preemptible_instance
from ..instances.preemptible.is_preemptible import is_preemptible
from ..instances.preemptible.preemption_history import list_zone_operations
PROJECT = google.auth.default()[1]
INSTANCE_ZONE = 'europe-west2-c'

@pytest.fixture
def autodelete_instance_name():
    if False:
        while True:
            i = 10
    instance_name = 'i' + uuid.uuid4().hex[:10]
    yield instance_name
    delete_instance(PROJECT, INSTANCE_ZONE, instance_name)

def test_preemptible_creation(autodelete_instance_name):
    if False:
        i = 10
        return i + 15
    instance = create_preemptible_instance(PROJECT, INSTANCE_ZONE, autodelete_instance_name)
    assert instance.name == autodelete_instance_name
    assert is_preemptible(PROJECT, INSTANCE_ZONE, instance.name)
    operations = list_zone_operations(PROJECT, INSTANCE_ZONE, f'targetLink="https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{INSTANCE_ZONE}/instances/{instance.name}"')
    try:
        next(iter(operations))
    except StopIteration:
        pytest.fail('There should be at least one operation for this instance at this point.')