from __future__ import annotations
import os
from datetime import datetime
import yandex.cloud.dataproc.v1.cluster_pb2 as cluster_pb
import yandex.cloud.dataproc.v1.cluster_service_pb2 as cluster_service_pb
import yandex.cloud.dataproc.v1.cluster_service_pb2_grpc as cluster_service_grpc_pb
import yandex.cloud.dataproc.v1.common_pb2 as common_pb
import yandex.cloud.dataproc.v1.job_pb2 as job_pb
import yandex.cloud.dataproc.v1.job_service_pb2 as job_service_pb
import yandex.cloud.dataproc.v1.job_service_pb2_grpc as job_service_grpc_pb
import yandex.cloud.dataproc.v1.subcluster_pb2 as subcluster_pb
from google.protobuf.json_format import MessageToDict
from airflow import DAG
from airflow.decorators import task
from airflow.providers.yandex.hooks.yandex import YandexCloudBaseHook
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'example_yandexcloud_hook'
YC_S3_BUCKET_NAME = ''
YC_FOLDER_ID = None
YC_ZONE_NAME = 'ru-central1-b'
YC_SUBNET_ID = None
YC_SERVICE_ACCOUNT_ID = None

def create_cluster_request(folder_id: str, cluster_name: str, cluster_desc: str, zone: str, subnet_id: str, service_account_id: str, ssh_public_key: str, resources: common_pb.Resources):
    if False:
        print('Hello World!')
    return cluster_service_pb.CreateClusterRequest(folder_id=folder_id, name=cluster_name, description=cluster_desc, bucket=YC_S3_BUCKET_NAME, config_spec=cluster_service_pb.CreateClusterConfigSpec(hadoop=cluster_pb.HadoopConfig(services=('SPARK', 'YARN'), ssh_public_keys=[ssh_public_key]), subclusters_spec=[cluster_service_pb.CreateSubclusterConfigSpec(name='master', role=subcluster_pb.Role.MASTERNODE, resources=resources, subnet_id=subnet_id, hosts_count=1), cluster_service_pb.CreateSubclusterConfigSpec(name='compute', role=subcluster_pb.Role.COMPUTENODE, resources=resources, subnet_id=subnet_id, hosts_count=1)]), zone_id=zone, service_account_id=service_account_id)

@task
def create_cluster(yandex_conn_id: str | None=None, folder_id: str | None=None, network_id: str | None=None, subnet_id: str | None=None, zone: str=YC_ZONE_NAME, service_account_id: str | None=None, ssh_public_key: str | None=None, *, dag: DAG | None=None, ts_nodash: str | None=None) -> str:
    if False:
        return 10
    hook = YandexCloudBaseHook(yandex_conn_id=yandex_conn_id)
    folder_id = folder_id or hook.default_folder_id
    if subnet_id is None:
        network_id = network_id or hook.sdk.helpers.find_network_id(folder_id)
        subnet_id = hook.sdk.helpers.find_subnet_id(folder_id=folder_id, zone_id=zone, network_id=network_id)
    service_account_id = service_account_id or hook.sdk.helpers.find_service_account_id()
    ssh_public_key = ssh_public_key or hook.default_public_ssh_key
    dag_id = dag and dag.dag_id or 'dag'
    request = create_cluster_request(folder_id=folder_id, subnet_id=subnet_id, zone=zone, cluster_name=f'airflow_{dag_id}_{ts_nodash}'[:62], cluster_desc='Created via Airflow custom hook task', service_account_id=service_account_id, ssh_public_key=ssh_public_key, resources=common_pb.Resources(resource_preset_id='s2.micro', disk_type_id='network-ssd'))
    operation = hook.sdk.client(cluster_service_grpc_pb.ClusterServiceStub).Create(request)
    operation_result = hook.sdk.wait_operation_and_get_result(operation, response_type=cluster_pb.Cluster, meta_type=cluster_service_pb.CreateClusterMetadata)
    return operation_result.response.id

@task
def run_spark_job(cluster_id: str, yandex_conn_id: str | None=None):
    if False:
        for i in range(10):
            print('nop')
    hook = YandexCloudBaseHook(yandex_conn_id=yandex_conn_id)
    request = job_service_pb.CreateJobRequest(cluster_id=cluster_id, name='Spark job: Find total urban population in distribution by country', spark_job=job_pb.SparkJob(main_jar_file_uri='file:///usr/lib/spark/examples/jars/spark-examples.jar', main_class='org.apache.spark.examples.SparkPi', args=['1000']))
    operation = hook.sdk.client(job_service_grpc_pb.JobServiceStub).Create(request)
    operation_result = hook.sdk.wait_operation_and_get_result(operation, response_type=job_pb.Job, meta_type=job_service_pb.CreateJobMetadata)
    return MessageToDict(operation_result.response)

@task(trigger_rule='all_done')
def delete_cluster(cluster_id: str, yandex_conn_id: str | None=None):
    if False:
        while True:
            i = 10
    hook = YandexCloudBaseHook(yandex_conn_id=yandex_conn_id)
    operation = hook.sdk.client(cluster_service_grpc_pb.ClusterServiceStub).Delete(cluster_service_pb.DeleteClusterRequest(cluster_id=cluster_id))
    hook.sdk.wait_operation_and_get_result(operation, meta_type=cluster_service_pb.DeleteClusterMetadata)
with DAG(dag_id=DAG_ID, schedule=None, start_date=datetime(2021, 1, 1), tags=['example']) as dag:
    cluster_id = create_cluster(folder_id=YC_FOLDER_ID, subnet_id=YC_SUBNET_ID, zone=YC_ZONE_NAME, service_account_id=YC_SERVICE_ACCOUNT_ID)
    spark_job = run_spark_job(cluster_id=cluster_id)
    delete_task = delete_cluster(cluster_id=cluster_id)
    spark_job >> delete_task
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)