from google.cloud import batch_v1

def create_script_job(project_id: str, region: str, job_name: str) -> batch_v1.Job:
    if False:
        for i in range(10):
            print('nop')
    '\n    This method shows how to create a sample Batch Job that will run\n    a simple command on Cloud Compute instances.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        region: name of the region you want to use to run the job. Regions that are\n            available for Batch are listed on: https://cloud.google.com/batch/docs/get-started#locations\n        job_name: the name of the job that will be created.\n            It needs to be unique for each project and region pair.\n\n    Returns:\n        A job object representing the job created.\n    '
    client = batch_v1.BatchServiceClient()
    task = batch_v1.TaskSpec()
    runnable = batch_v1.Runnable()
    runnable.script = batch_v1.Runnable.Script()
    runnable.script.text = 'echo Hello world! This is task ${BATCH_TASK_INDEX}. This job has a total of ${BATCH_TASK_COUNT} tasks.'
    task.runnables = [runnable]
    resources = batch_v1.ComputeResource()
    resources.cpu_milli = 2000
    resources.memory_mib = 16
    task.compute_resource = resources
    task.max_retry_count = 2
    task.max_run_duration = '3600s'
    group = batch_v1.TaskGroup()
    group.task_count = 4
    group.task_spec = task
    allocation_policy = batch_v1.AllocationPolicy()
    policy = batch_v1.AllocationPolicy.InstancePolicy()
    policy.machine_type = 'e2-standard-4'
    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
    allocation_policy.instances = [instances]
    job = batch_v1.Job()
    job.task_groups = [group]
    job.allocation_policy = allocation_policy
    job.labels = {'env': 'testing', 'type': 'script'}
    job.logs_policy = batch_v1.LogsPolicy()
    job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
    create_request = batch_v1.CreateJobRequest()
    create_request.job = job
    create_request.job_id = job_name
    create_request.parent = f'projects/{project_id}/locations/{region}'
    return client.create_job(create_request)