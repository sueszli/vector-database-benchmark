import urllib
from google.cloud import tasks_v2beta3 as tasks

def create_http_queue(project: str, location: str, name: str, uri: str) -> tasks.Queue:
    if False:
        for i in range(10):
            print('nop')
    "Create an HTTP queue.\n    Args:\n        project: The project ID to create the queue in.\n        location: The location to create the queue in.\n        name: The ID to use for the new queue.\n        uri: The HTTP endpoint's URI for all tasks in the queue\n\n    Returns:\n        The newly created queue.\n    "
    client = tasks.CloudTasksClient()
    parsedUri = urllib.parse.urlparse(uri)
    http_target = {'uri_override': {'host': parsedUri.hostname, 'uri_override_enforce_mode': tasks.types.UriOverride.UriOverrideEnforceMode.ALWAYS}}
    if parsedUri.scheme == 'http':
        http_target['uri_override']['scheme'] = tasks.types.UriOverride.Scheme.HTTP
    if parsedUri.port:
        http_target['uri_override']['port'] = f'{parsedUri.port}'
    if parsedUri.path:
        http_target['uri_override']['path_override'] = {'path': parsedUri.path}
    if parsedUri.query:
        http_target['uri_override']['query_override'] = {'query_params': parsedUri.query}
    queue = client.create_queue(tasks.CreateQueueRequest(parent=client.common_location_path(project, location), queue={'name': f'projects/{project}/locations/{location}/queues/{name}', 'http_target': http_target}))
    return queue