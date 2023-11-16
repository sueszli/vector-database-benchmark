from google.cloud import tasks_v2beta3 as tasks

def delete_http_queue(queue: tasks.Queue) -> None:
    if False:
        i = 10
        return i + 15
    'Delete an HTTP queue.\n    Args:\n        queue: The queue to delete.\n    Returns:\n        None.\n    '
    client = tasks.CloudTasksClient()
    client.delete_queue(name=queue.name)