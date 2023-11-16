import time
import uuid
import google.auth
from google.cloud import compute_v1
import pytest
from ..images.get import get_image_from_family
from ..instances.create import create_instance, disk_from_image
from ..instances.delete import delete_instance
from ..instances.resume import resume_instance
from ..instances.suspend import suspend_instance
PROJECT = google.auth.default()[1]
INSTANCE_ZONE = 'europe-west2-b'

def _get_status(instance: compute_v1.Instance) -> compute_v1.Instance.Status:
    if False:
        print('Hello World!')
    instance_client = compute_v1.InstancesClient()
    return instance_client.get(project=PROJECT, zone=INSTANCE_ZONE, instance=instance.name).status

@pytest.fixture
def compute_instance():
    if False:
        i = 10
        return i + 15
    instance_name = 'test-instance-' + uuid.uuid4().hex[:10]
    newest_debian = get_image_from_family(project='ubuntu-os-cloud', family='ubuntu-2004-lts')
    disk_type = f'zones/{INSTANCE_ZONE}/diskTypes/pd-standard'
    disks = [disk_from_image(disk_type, 100, True, newest_debian.self_link)]
    instance = create_instance(PROJECT, INSTANCE_ZONE, instance_name, disks)
    yield instance
    delete_instance(PROJECT, INSTANCE_ZONE, instance_name)

def test_instance_suspend_resume(compute_instance):
    if False:
        print('Hello World!')
    assert _get_status(compute_instance) == compute_v1.Instance.Status.RUNNING.name
    time.sleep(45)
    suspend_instance(PROJECT, INSTANCE_ZONE, compute_instance.name)
    while _get_status(compute_instance) == compute_v1.Instance.Status.SUSPENDING.name:
        time.sleep(5)
    assert _get_status(compute_instance) == compute_v1.Instance.Status.SUSPENDED.name
    resume_instance(PROJECT, INSTANCE_ZONE, compute_instance.name)
    assert _get_status(compute_instance) == compute_v1.Instance.Status.RUNNING.name