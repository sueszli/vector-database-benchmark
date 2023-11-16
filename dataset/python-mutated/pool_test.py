import os
import pytest
import get_pool
project_name = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us-central1'
pool_id = 'default'
peered_network = ''

def test_pool_operations(capsys: pytest.fixture) -> None:
    if False:
        while True:
            i = 10
    pool_name_project_id = f'projects/{project_name}/locations/{location}/pools/{pool_id}'
    response = get_pool.get_pool(project_name, location, pool_id)
    assert pool_name_project_id in response.name