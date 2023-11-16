from __future__ import annotations
from collections.abc import Generator
import os
import pathlib
import re
import uuid
from google.api_core.exceptions import NotFound
from google.cloud import dataproc_v1, storage
from google.cloud.dataproc_v1.types import LoggingConfig
from google.cloud.pubsublite import AdminClient, Subscription, Topic
from google.cloud.pubsublite.types import BacklogLocation, CloudRegion, CloudZone, SubscriptionPath, TopicPath
import pytest
UUID = uuid.uuid4().hex
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
PROJECT_NUMBER = os.environ['GOOGLE_CLOUD_PROJECT_NUMBER']
CLOUD_REGION = 'us-west1'
ZONE_ID = 'a'
BUCKET = os.environ['PUBSUBLITE_BUCKET_ID']
CLUSTER_ID = os.environ['PUBSUBLITE_CLUSTER_ID'] + '-' + UUID
TOPIC_ID = 'spark-streaming-topic-' + UUID
SUBSCRIPTION_ID = 'spark-streaming-subscription-' + UUID
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

@pytest.fixture(scope='module')
def client() -> Generator[AdminClient, None, None]:
    if False:
        print('Hello World!')
    yield AdminClient(CLOUD_REGION)

@pytest.fixture(scope='module')
def topic(client: AdminClient) -> Generator[Topic, None, None]:
    if False:
        return 10
    location = CloudZone(CloudRegion(CLOUD_REGION), ZONE_ID)
    topic_path = TopicPath(PROJECT_NUMBER, location, TOPIC_ID)
    topic = Topic(name=str(topic_path), partition_config=Topic.PartitionConfig(count=2, capacity=Topic.PartitionConfig.Capacity(publish_mib_per_sec=4, subscribe_mib_per_sec=8)), retention_config=Topic.RetentionConfig(per_partition_bytes=30 * 1024 * 1024 * 1024))
    try:
        response = client.get_topic(topic.name)
    except NotFound:
        response = client.create_topic(topic)
    yield response
    try:
        client.delete_topic(response.name)
    except NotFound as e:
        print(e.message)

@pytest.fixture(scope='module')
def subscription(client: AdminClient, topic: Topic) -> Generator[Subscription, None, None]:
    if False:
        for i in range(10):
            print('nop')
    location = CloudZone(CloudRegion(CLOUD_REGION), ZONE_ID)
    subscription_path = SubscriptionPath(PROJECT_NUMBER, location, SUBSCRIPTION_ID)
    subscription = Subscription(name=str(subscription_path), topic=topic.name, delivery_config=Subscription.DeliveryConfig(delivery_requirement=Subscription.DeliveryConfig.DeliveryRequirement.DELIVER_IMMEDIATELY))
    try:
        response = client.get_subscription(subscription.name)
    except NotFound:
        response = client.create_subscription(subscription, BacklogLocation.BEGINNING)
    yield response
    try:
        client.delete_subscription(response.name)
    except NotFound as e:
        print(e.message)

@pytest.fixture(scope='module')
def dataproc_cluster() -> Generator[dataproc_v1.Cluster, None, None]:
    if False:
        for i in range(10):
            print('nop')
    cluster_client = dataproc_v1.ClusterControllerClient(client_options={'api_endpoint': f'{CLOUD_REGION}-dataproc.googleapis.com:443'})
    cluster = {'project_id': PROJECT_ID, 'cluster_name': CLUSTER_ID, 'config': {'master_config': {'num_instances': 1, 'machine_type_uri': 'n1-standard-2', 'disk_config': {'boot_disk_size_gb': 100}}, 'worker_config': {'num_instances': 2, 'machine_type_uri': 'n1-standard-2', 'disk_config': {'boot_disk_size_gb': 100}}, 'config_bucket': BUCKET, 'temp_bucket': BUCKET, 'software_config': {'image_version': '2.0-debian10'}, 'gce_cluster_config': {'service_account_scopes': ['https://www.googleapis.com/auth/cloud-platform']}, 'lifecycle_config': {'idle_delete_ttl': {'seconds': 3600}}}}
    operation = cluster_client.create_cluster(request={'project_id': PROJECT_ID, 'region': CLOUD_REGION, 'cluster': cluster})
    result = operation.result()
    yield result
    cluster_client.delete_cluster(request={'project_id': PROJECT_ID, 'region': CLOUD_REGION, 'cluster_name': result.cluster_name})

def pyfile(source_file: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET)
    destination_blob_name = os.path.join(UUID, source_file)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file)
    return 'gs://' + blob.bucket.name + '/' + blob.name

def test_spark_streaming_to_pubsublite(topic: Topic, dataproc_cluster: dataproc_v1.Cluster) -> None:
    if False:
        i = 10
        return i + 15
    job_client = dataproc_v1.JobControllerClient(client_options={'api_endpoint': f'{CLOUD_REGION}-dataproc.googleapis.com:443'})
    job = {'reference': {'job_id': topic.name.split('/')[-1][:-28]}, 'placement': {'cluster_name': dataproc_cluster.cluster_name}, 'pyspark_job': {'main_python_file_uri': pyfile('spark_streaming_to_pubsublite_example.py'), 'jar_file_uris': ['gs://pubsublite-spark/pubsublite-spark-sql-streaming-1.0.0-with-dependencies.jar'], 'properties': {'spark.master': 'yarn'}, 'logging_config': {'driver_log_levels': {'root': LoggingConfig.Level.INFO}}, 'args': [f'--project_number={PROJECT_NUMBER}', f'--location={CLOUD_REGION}-{ZONE_ID}', f'--topic_id={TOPIC_ID}']}}
    operation = job_client.submit_job_as_operation(request={'project_id': PROJECT_ID, 'region': CLOUD_REGION, 'job': job, 'request_id': 'write-' + UUID})
    response = operation.result()
    matches = re.match('gs://(.*?)/(.*)', response.driver_output_resource_uri)
    output = storage.Client().get_bucket(matches.group(1)).blob(f'{matches.group(2)}.000000000').download_as_text()
    assert 'Committed 1 messages for epochId' in output

def test_spark_streaming_from_pubsublite(subscription: Subscription, dataproc_cluster: dataproc_v1.Cluster) -> None:
    if False:
        print('Hello World!')
    job_client = dataproc_v1.JobControllerClient(client_options={'api_endpoint': f'{CLOUD_REGION}-dataproc.googleapis.com:443'})
    job = {'reference': {'job_id': subscription.name.split('/')[-1][:-28]}, 'placement': {'cluster_name': dataproc_cluster.cluster_name}, 'pyspark_job': {'main_python_file_uri': pyfile('spark_streaming_from_pubsublite_example.py'), 'jar_file_uris': ['gs://spark-lib/pubsublite/pubsublite-spark-sql-streaming-LATEST-with-dependencies.jar'], 'properties': {'spark.master': 'yarn'}, 'logging_config': {'driver_log_levels': {'root': LoggingConfig.Level.INFO}}, 'args': [f'--project_number={PROJECT_NUMBER}', f'--location={CLOUD_REGION}-{ZONE_ID}', f'--subscription_id={SUBSCRIPTION_ID}']}}
    operation = job_client.submit_job_as_operation(request={'project_id': PROJECT_ID, 'region': CLOUD_REGION, 'job': job, 'request_id': 'read-' + UUID})
    response = operation.result()
    matches = re.match('gs://(.*?)/(.*)', response.driver_output_resource_uri)
    output = storage.Client().get_bucket(matches.group(1)).blob(f'{matches.group(2)}.000000000').download_as_text()
    assert 'Batch: 0\n' in output