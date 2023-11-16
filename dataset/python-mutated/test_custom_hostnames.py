import random
import uuid
import google.auth
import pytest
from ..instances.custom_hostname.create import create_instance_custom_hostname
from ..instances.custom_hostname.get import get_hostname
from ..instances.delete import delete_instance
PROJECT = google.auth.default()[1]
INSTANCE_ZONE = 'europe-west1-c'

@pytest.fixture
def autodelete_instance_name():
    if False:
        print('Hello World!')
    instance_name = 'test-host-instance-' + uuid.uuid4().hex[:10]
    yield instance_name
    delete_instance(PROJECT, INSTANCE_ZONE, instance_name)

@pytest.fixture
def random_hostname():
    if False:
        return 10
    yield 'instance.{}.hostname'.format(random.randint(0, 2 ** 10))

def test_custom_hostname(autodelete_instance_name, random_hostname):
    if False:
        for i in range(10):
            print('nop')
    instance = create_instance_custom_hostname(PROJECT, INSTANCE_ZONE, autodelete_instance_name, random_hostname)
    assert instance.name == autodelete_instance_name
    assert instance.hostname == random_hostname
    assert get_hostname(PROJECT, INSTANCE_ZONE, autodelete_instance_name) == random_hostname