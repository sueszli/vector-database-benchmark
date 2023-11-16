import argparse
import datetime
import json
from typing import Dict, Optional
from google.cloud import tasks_v2
from google.protobuf import duration_pb2, timestamp_pb2

def create_http_task(project: str, location: str, queue: str, url: str, json_payload: Dict, scheduled_seconds_from_now: Optional[int]=None, task_id: Optional[str]=None, deadline_in_seconds: Optional[int]=None) -> tasks_v2.Task:
    if False:
        i = 10
        return i + 15
    'Create an HTTP POST task with a JSON payload.\n    Args:\n        project: The project ID where the queue is located.\n        location: The location where the queue is located.\n        queue: The ID of the queue to add the task to.\n        url: The target URL of the task.\n        json_payload: The JSON payload to send.\n        scheduled_seconds_from_now: Seconds from now to schedule the task for.\n        task_id: ID to use for the newly created task.\n        deadline_in_seconds: The deadline in seconds for task.\n    Returns:\n        The newly created task.\n    '
    client = tasks_v2.CloudTasksClient()
    task = tasks_v2.Task(http_request=tasks_v2.HttpRequest(http_method=tasks_v2.HttpMethod.POST, url=url, headers={'Content-type': 'application/json'}, body=json.dumps(json_payload).encode()), name=client.task_path(project, location, queue, task_id) if task_id is not None else None)
    if scheduled_seconds_from_now is not None:
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(datetime.datetime.utcnow() + datetime.timedelta(seconds=scheduled_seconds_from_now))
        task.schedule_time = timestamp
    if deadline_in_seconds is not None:
        duration = duration_pb2.Duration()
        duration.FromSeconds(deadline_in_seconds)
        task.dispatch_deadline = duration
    return client.create_task(tasks_v2.CreateTaskRequest(parent=client.queue_path(project, location, queue), task=task))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=create_http_task.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', help='Project of the queue to add the task to.', required=True)
    parser.add_argument('--queue', help='ID (short name) of the queue to add the task to.', required=True)
    parser.add_argument('--location', help='Location of the queue to add the task to.', required=True)
    parser.add_argument('--url', help='The full url path that the request will be sent to.', required=True)
    parser.add_argument('--payload', help='Optional payload to attach to the push queue.')
    parser.add_argument('--in_seconds', type=int, help='The number of seconds from now to schedule task attempt.')
    parser.add_argument('--task_name', help='Task name of the task to create')
    args = parser.parse_args()
    create_http_task(args.project, args.queue, args.location, args.url, args.payload, args.in_seconds, args.task_name)