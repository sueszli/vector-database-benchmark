import uuid
import google.auth
import pytest
from ..instances.delete import delete_instance
from ..instances.delete_protection.create import create_protected_instance
from ..instances.delete_protection.get import get_delete_protection
from ..instances.delete_protection.set import set_delete_protection
PROJECT = google.auth.default()[1]
INSTANCE_ZONE = 'europe-west2-a'

@pytest.fixture
def autodelete_instance_name():
    if False:
        return 10
    instance_name = 'test-instance-' + uuid.uuid4().hex[:10]
    yield instance_name
    if get_delete_protection(PROJECT, INSTANCE_ZONE, instance_name):
        set_delete_protection(PROJECT, INSTANCE_ZONE, instance_name, False)
    delete_instance(PROJECT, INSTANCE_ZONE, instance_name)

def test_delete_protection(autodelete_instance_name):
    if False:
        while True:
            i = 10
    instance = create_protected_instance(PROJECT, INSTANCE_ZONE, autodelete_instance_name)
    assert instance.name == autodelete_instance_name
    assert get_delete_protection(PROJECT, INSTANCE_ZONE, autodelete_instance_name) is True
    set_delete_protection(PROJECT, INSTANCE_ZONE, autodelete_instance_name, False)
    assert get_delete_protection(PROJECT, INSTANCE_ZONE, autodelete_instance_name) is False