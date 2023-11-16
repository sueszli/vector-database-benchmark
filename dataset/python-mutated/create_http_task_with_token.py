from typing import Optional
from google.cloud import tasks_v2

def create_http_task_with_token(project: str, location: str, queue: str, url: str, payload: bytes, service_account_email: str, audience: Optional[str]=None) -> tasks_v2.Task:
    if False:
        for i in range(10):
            print('nop')
    'Create an HTTP POST task with an OIDC token and an arbitrary payload.\n    Args:\n        project: The project ID where the queue is located.\n        location: The location where the queue is located.\n        queue: The ID of the queue to add the task to.\n        url: The target URL of the task.\n        payload: The payload to send.\n        service_account_email: The service account to use for generating the OIDC token.\n        audience: Audience to use when generating the OIDC token.\n    Returns:\n        The newly created task.\n    '
    client = tasks_v2.CloudTasksClient()
    task = tasks_v2.Task(http_request=tasks_v2.HttpRequest(http_method=tasks_v2.HttpMethod.POST, url=url, oidc_token=tasks_v2.OidcToken(service_account_email=service_account_email, audience=audience), body=payload))
    return client.create_task(tasks_v2.CreateTaskRequest(parent=client.queue_path(project, location, queue), task=task))