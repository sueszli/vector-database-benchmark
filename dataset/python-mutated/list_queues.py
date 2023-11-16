from typing import List
from google.cloud import tasks_v2

def list_queues(project: str, location: str) -> List[str]:
    if False:
        return 10
    'List all queues\n    Args:\n        project: The project ID to list queues from.\n        location: The location ID to list queues from.\n\n    Returns:\n        A list of queue names.\n    '
    client = tasks_v2.CloudTasksClient()
    response = client.list_queues(tasks_v2.ListQueuesRequest(parent=client.common_location_path(project, location)))
    return [queue.name for queue in response]