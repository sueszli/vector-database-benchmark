import os
import uuid
import backoff
from google.api_core.exceptions import AlreadyExists, Cancelled, InternalServerError, InvalidArgument, NotFound, ServiceUnavailable
from google.cloud.dataproc_v1 import ClusterStatus, GetClusterRequest
from google.cloud.dataproc_v1.services.cluster_controller.client import ClusterControllerClient
import pytest
import update_cluster
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
REGION = 'us-central1'
CLUSTER_NAME = f'py-cc-test-{str(uuid.uuid4())}'
NEW_NUM_INSTANCES = 3
CLUSTER = {'project_id': PROJECT_ID, 'cluster_name': CLUSTER_NAME, 'config': {'master_config': {'num_instances': 1, 'machine_type_uri': 'n1-standard-2', 'disk_config': {'boot_disk_size_gb': 100}}, 'worker_config': {'num_instances': 2, 'machine_type_uri': 'n1-standard-2', 'disk_config': {'boot_disk_size_gb': 100}}}}

@pytest.fixture(scope='module')
def cluster_client():
    if False:
        return 10
    cluster_client = ClusterControllerClient(client_options={'api_endpoint': f'{REGION}-dataproc.googleapis.com:443'})
    return cluster_client

@backoff.on_exception(backoff.expo, (ServiceUnavailable, InvalidArgument), max_tries=5)
def setup_cluster(cluster_client):
    if False:
        for i in range(10):
            print('nop')
    try:
        operation = cluster_client.create_cluster(request={'project_id': PROJECT_ID, 'region': REGION, 'cluster': CLUSTER})
        operation.result()
    except AlreadyExists:
        print('Cluster already exists, utilize existing cluster')

@backoff.on_exception(backoff.expo, ServiceUnavailable, max_tries=5)
def teardown_cluster(cluster_client):
    if False:
        while True:
            i = 10
    try:
        operation = cluster_client.delete_cluster(request={'project_id': PROJECT_ID, 'region': REGION, 'cluster_name': CLUSTER_NAME})
        operation.result()
    except NotFound:
        print('Cluster already deleted')

@backoff.on_exception(backoff.expo, (InternalServerError, ServiceUnavailable, Cancelled), max_tries=5)
def test_update_cluster(capsys, cluster_client: ClusterControllerClient):
    if False:
        return 10

    def test_update_cluster_inner(cluster_client: ClusterControllerClient, update_retries: int):
        if False:
            for i in range(10):
                print('nop')
        try:
            setup_cluster(cluster_client)
            request = GetClusterRequest(project_id=PROJECT_ID, region=REGION, cluster_name=CLUSTER_NAME)
            response = cluster_client.get_cluster(request=request)
            assert response.status.state == ClusterStatus.State.RUNNING
            update_cluster.update_cluster(PROJECT_ID, REGION, CLUSTER_NAME, NEW_NUM_INSTANCES)
            new_num_cluster = cluster_client.get_cluster(project_id=PROJECT_ID, region=REGION, cluster_name=CLUSTER_NAME)
            (out, _) = capsys.readouterr()
            assert CLUSTER_NAME in out
            assert new_num_cluster.config.worker_config.num_instances == NEW_NUM_INSTANCES
        except AssertionError as e:
            if update_retries < 3 and response.status.state == ClusterStatus.State.ERROR:
                teardown_cluster(cluster_client)
                test_update_cluster_inner(cluster_client=cluster_client, update_retries=update_retries + 1)
            else:
                raise e
        finally:
            teardown_cluster(cluster_client)
    test_update_cluster_inner(cluster_client=cluster_client, update_retries=0)