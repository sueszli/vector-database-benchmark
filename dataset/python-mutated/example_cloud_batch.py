"""
Example Airflow DAG that uses Google Cloud Batch Operators.
"""
from __future__ import annotations
import os
from datetime import datetime
from google.cloud import batch_v1
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.cloud_batch import CloudBatchDeleteJobOperator, CloudBatchListJobsOperator, CloudBatchListTasksOperator, CloudBatchSubmitJobOperator
from airflow.utils.trigger_rule import TriggerRule
PROJECT_ID = os.environ.get('SYSTEM_TESTS_GCP_PROJECT', 'default')
DAG_ID = 'example_cloud_batch'
region = 'us-central1'
job_name_prefix = 'batch-system-test-job'
job1_name = f'{job_name_prefix}1'
job2_name = f'{job_name_prefix}2'
submit1_task_name = 'submit-job1'
submit2_task_name = 'submit-job2'
delete1_task_name = 'delete-job1'
delete2_task_name = 'delete-job2'
list_jobs_task_name = 'list-jobs'
list_tasks_task_name = 'list-tasks'
clean1_task_name = 'clean-job1'
clean2_task_name = 'clean-job2'

def _assert_jobs(ti):
    if False:
        i = 10
        return i + 15
    job_names = ti.xcom_pull(task_ids=[list_jobs_task_name], key='return_value')
    job_names_str = job_names[0][0]['name'].split('/')[-1] + ' ' + job_names[0][1]['name'].split('/')[-1]
    assert job1_name in job_names_str
    assert job2_name in job_names_str

def _assert_tasks(ti):
    if False:
        return 10
    tasks_names = ti.xcom_pull(task_ids=[list_tasks_task_name], key='return_value')
    assert len(tasks_names[0]) == 2
    assert 'tasks/0' in tasks_names[0][0]['name']
    assert 'tasks/1' in tasks_names[0][1]['name']

def _create_job():
    if False:
        i = 10
        return i + 15
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.image_uri = 'gcr.io/google-containers/busybox'
    runnable.container.entrypoint = '/bin/sh'
    runnable.container.commands = ['-c', 'echo Hello world! This is task ${BATCH_TASK_INDEX}.          This job has a total of ${BATCH_TASK_COUNT} tasks.']
    task = batch_v1.TaskSpec()
    task.runnables = [runnable]
    resources = batch_v1.ComputeResource()
    resources.cpu_milli = 2000
    resources.memory_mib = 16
    task.compute_resource = resources
    task.max_retry_count = 2
    group = batch_v1.TaskGroup()
    group.task_count = 2
    group.task_spec = task
    policy = batch_v1.AllocationPolicy.InstancePolicy()
    policy.machine_type = 'e2-standard-4'
    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
    allocation_policy = batch_v1.AllocationPolicy()
    allocation_policy.instances = [instances]
    job = batch_v1.Job()
    job.task_groups = [group]
    job.allocation_policy = allocation_policy
    job.labels = {'env': 'testing', 'type': 'container'}
    job.logs_policy = batch_v1.LogsPolicy()
    job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
    return job
with DAG(DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['example', 'batch']) as dag:
    submit1 = CloudBatchSubmitJobOperator(task_id=submit1_task_name, project_id=PROJECT_ID, region=region, job_name=job1_name, job=_create_job(), dag=dag, deferrable=False)
    submit2 = CloudBatchSubmitJobOperator(task_id=submit2_task_name, project_id=PROJECT_ID, region=region, job_name=job2_name, job=batch_v1.Job.to_dict(_create_job()), dag=dag, deferrable=True)
    list_tasks = CloudBatchListTasksOperator(task_id=list_tasks_task_name, project_id=PROJECT_ID, region=region, job_name=job1_name, dag=dag)
    assert_tasks = PythonOperator(task_id='assert-tasks', python_callable=_assert_tasks, dag=dag)
    list_jobs = CloudBatchListJobsOperator(task_id=list_jobs_task_name, project_id=PROJECT_ID, region=region, limit=2, filter=f'name:projects/{PROJECT_ID}/locations/{region}/jobs/{job_name_prefix}*', dag=dag)
    get_name = PythonOperator(task_id='assert-jobs', python_callable=_assert_jobs, dag=dag)
    delete_job1 = CloudBatchDeleteJobOperator(task_id='delete-job1', project_id=PROJECT_ID, region=region, job_name=job1_name, dag=dag, trigger_rule=TriggerRule.ALL_DONE)
    delete_job2 = CloudBatchDeleteJobOperator(task_id='delete-job2', project_id=PROJECT_ID, region=region, job_name=job2_name, dag=dag, trigger_rule=TriggerRule.ALL_DONE)
    [submit1, submit2] >> list_tasks >> assert_tasks >> list_jobs >> get_name >> [delete_job1, delete_job2]
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)