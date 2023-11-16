from google.cloud import tasks_v2

def delete_queue(project: str, location: str, queue_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Delete a queue.\n    Args:\n        project: The project ID where the queue is located.\n        location: The location ID where the queue is located.\n        queue_id: The ID of the queue to delete.\n    '
    client = tasks_v2.CloudTasksClient()
    client.delete_queue(tasks_v2.DeleteQueueRequest(name=client.queue_path(project, location, queue_id)))