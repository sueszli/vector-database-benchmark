from google.cloud import tasks_v2

def create_queue(project: str, location: str, queue_id: str) -> tasks_v2.Queue:
    if False:
        while True:
            i = 10
    'Create a queue.\n    Args:\n        project: The project ID to create the queue in.\n        location: The location to create the queue in.\n        queue_id: The ID to use for the new queue.\n\n    Returns:\n        The newly created queue.\n    '
    client = tasks_v2.CloudTasksClient()
    return client.create_queue(tasks_v2.CreateQueueRequest(parent=client.common_location_path(project, location), queue=tasks_v2.Queue(name=client.queue_path(project, location, queue_id))))