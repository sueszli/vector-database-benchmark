import os
import uuid
import backoff
from google.api_core import exceptions as googleEx
from google.cloud import container_v1 as gke
import pytest
import delete_cluster as gke_delete
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
ZONE = 'us-central1-b'
CLUSTER_NAME = f'py-container-repo-test-{uuid.uuid4().hex[:10]}'

@pytest.fixture(autouse=True)
def setup_and_tear_down() -> None:
    if False:
        return 10
    client = gke.ClusterManagerClient()
    cluster_location = client.common_location_path(PROJECT_ID, ZONE)
    cluster_def = {'name': CLUSTER_NAME, 'initial_node_count': 2, 'node_config': {'machine_type': 'e2-standard-2'}}
    op = client.create_cluster({'parent': cluster_location, 'cluster': cluster_def})
    op_id = f'{cluster_location}/operations/{op.name}'

    @backoff.on_predicate(backoff.expo, lambda x: x != gke.Operation.Status.DONE, max_tries=20)
    def wait_for_create() -> gke.Operation.Status:
        if False:
            while True:
                i = 10
        return client.get_operation({'name': op_id}).status
    wait_for_create()
    yield
    client = gke.ClusterManagerClient()
    cluster_location = client.common_location_path(PROJECT_ID, ZONE)
    cluster_name = f'{cluster_location}/clusters/{CLUSTER_NAME}'
    try:
        op = client.delete_cluster({'name': cluster_name})
        op_id = f'{cluster_location}/operations/{op.name}'

        @backoff.on_predicate(backoff.expo, lambda x: x != gke.Operation.Status.DONE, max_tries=20)
        def wait_for_delete() -> gke.Operation.Status:
            if False:
                for i in range(10):
                    print('nop')
            return client.get_operation({'name': op_id}).status
        wait_for_delete()
    except googleEx.NotFound:
        pass

def test_delete_clusters(capsys: object) -> None:
    if False:
        for i in range(10):
            print('nop')
    gke_delete.delete_cluster(PROJECT_ID, ZONE, CLUSTER_NAME)
    (out, _) = capsys.readouterr()
    assert 'Backing off ' in out
    assert 'Successfully deleted cluster after' in out
    client = gke.ClusterManagerClient()
    cluster_location = client.common_location_path(PROJECT_ID, ZONE)
    list_response = client.list_clusters({'parent': cluster_location})
    list_of_clusters = []
    for cluster in list_response.clusters:
        list_of_clusters.append(cluster.name)
    assert CLUSTER_NAME not in list_of_clusters