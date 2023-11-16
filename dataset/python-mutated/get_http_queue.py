from google.cloud import tasks_v2beta3 as tasks

def get_http_queue(project: str, location: str, name: str) -> tasks.Queue:
    if False:
        for i in range(10):
            print('nop')
    'Get an HTTP queue.\n    Args:\n        project: The project ID containing the queue.\n        location: The location containing the queue.\n        name: The ID of the queue.\n\n    Returns:\n        The matching queue, or None if it does not exist.\n    '
    client = tasks.CloudTasksClient()
    return client.get_queue(name=f'projects/{project}/locations/{location}/queues/{name}')