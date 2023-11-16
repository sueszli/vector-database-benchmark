from google.cloud import batch_v1

def get_task(project_id: str, region: str, job_name: str, group_name: str, task_number: int) -> batch_v1.Task:
    if False:
        i = 10
        return i + 15
    "\n    Retrieve information about a Task.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        region: name of the region hosts the job.\n        job_name: the name of the job you want to retrieve information about.\n        group_name: the name of the group that owns the task you want to check. Usually it's `group0`.\n        task_number: number of the task you want to look up.\n\n    Returns:\n        A Task object representing the specified task.\n    "
    client = batch_v1.BatchServiceClient()
    return client.get_task(name=f'projects/{project_id}/locations/{region}/jobs/{job_name}/taskGroups/{group_name}/tasks/{task_number}')