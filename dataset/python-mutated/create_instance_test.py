import os
import uuid
import pytest
from create_instance import main
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
BUCKET = os.environ['CLOUD_STORAGE_BUCKET']

@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_main(capsys):
    if False:
        for i in range(10):
            print('nop')
    instance_name = f'test-instance-{uuid.uuid4()}'
    main(PROJECT, BUCKET, 'europe-west1-b', instance_name, wait=False)
    (out, _) = capsys.readouterr()
    assert 'Instances in project' in out
    assert 'zone europe-west1-b' in out
    assert instance_name in out
    assert 'Deleting instance' in out