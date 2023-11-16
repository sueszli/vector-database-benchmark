from google.cloud import tasks_v2beta3 as tasks
import requests

def send_task_to_http_queue(queue: tasks.Queue, body: str='', token: str='', headers: dict={}) -> int:
    if False:
        i = 10
        return i + 15
    'Send a task to an HTTP queue.\n    Args:\n        queue: The queue to delete.\n        body: The body of the task.\n        auth_token: An authorization token for the queue.\n        headers: Headers to set on the task.\n    Returns:\n        The matching queue, or None if it does not exist.\n    '
    if token:
        headers['Authorization'] = f'Bearer {token}'
    endpoint = f'https://cloudtasks.googleapis.com/v2beta3/{queue.name}/tasks:buffer'
    response = requests.post(endpoint, body, headers=headers)
    return response.status_code