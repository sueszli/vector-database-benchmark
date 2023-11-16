import uuid
import google.auth
import pytest
from ..instances.delete import delete_instance
from ..instances.spot.create import create_spot_instance
from ..instances.spot.is_spot_vm import is_spot_vm
PROJECT = google.auth.default()[1]
INSTANCE_ZONE = 'europe-west2-c'

@pytest.fixture
def autodelete_instance_name():
    if False:
        return 10
    instance_name = 'i' + uuid.uuid4().hex[:10]
    yield instance_name
    delete_instance(PROJECT, INSTANCE_ZONE, instance_name)

def test_preemptible_creation(autodelete_instance_name):
    if False:
        i = 10
        return i + 15
    instance = create_spot_instance(PROJECT, INSTANCE_ZONE, autodelete_instance_name)
    assert instance.name == autodelete_instance_name
    assert is_spot_vm(PROJECT, INSTANCE_ZONE, instance.name)