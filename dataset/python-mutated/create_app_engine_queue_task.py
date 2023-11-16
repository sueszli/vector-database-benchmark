import argparse

def create_task(project, queue, location, payload=None, in_seconds=None):
    if False:
        print('Hello World!')
    'Create a task for a given queue with an arbitrary payload.'
    from google.cloud import tasks_v2
    from google.protobuf import timestamp_pb2
    import datetime
    import json
    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(project, location, queue)
    task = {'app_engine_http_request': {'http_method': tasks_v2.HttpMethod.POST, 'relative_uri': '/example_task_handler'}}
    if payload is not None:
        if isinstance(payload, dict):
            payload = json.dumps(payload)
            task['app_engine_http_request']['headers'] = {'Content-type': 'application/json'}
        converted_payload = payload.encode()
        task['app_engine_http_request']['body'] = converted_payload
    if in_seconds is not None:
        d = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=in_seconds)
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(d)
        task['schedule_time'] = timestamp
    response = client.create_task(parent=parent, task=task)
    print(f'Created task {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=create_task.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', help='Project of the queue to add the task to.', required=True)
    parser.add_argument('--queue', help='ID (short name) of the queue to add the task to.', required=True)
    parser.add_argument('--location', help='Location of the queue to add the task to.', required=True)
    parser.add_argument('--payload', help='Optional payload to attach to the push queue.')
    parser.add_argument('--in_seconds', type=int, help='The number of seconds from now to schedule task attempt.')
    args = parser.parse_args()
    create_task(args.project, args.queue, args.location, args.payload, args.in_seconds)