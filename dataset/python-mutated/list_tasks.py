from __future__ import annotations
from collections.abc import Iterable
from google.cloud import batch_v1

def list_tasks(project_id: str, region: str, job_name: str, group_name: str) -> Iterable[batch_v1.Task]:
    if False:
        while True:
            i = 10
    "\n    Get a list of all jobs defined in given region.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        region: name of the region hosting the jobs.\n        job_name: name of the job which tasks you want to list.\n        group_name: name of the group of tasks. Usually it's `group0`.\n\n    Returns:\n        An iterable collection of Task objects.\n    "
    client = batch_v1.BatchServiceClient()
    return client.list_tasks(parent=f'projects/{project_id}/locations/{region}/jobs/{job_name}/taskGroups/{group_name}')