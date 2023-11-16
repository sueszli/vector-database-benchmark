import urllib
from google.cloud import tasks_v2beta3 as tasks

def update_http_queue(queue: tasks.Queue, uri: str='', max_per_second: float=0.0, max_burst: int=0, max_concurrent: int=0, max_attempts: int=0) -> tasks.Queue:
    if False:
        print('Hello World!')
    'Update an HTTP queue with provided properties.\n    Args:\n        queue: The queue to update.\n        uri: The new HTTP endpoint\n        max_per_second: the new maximum number of dispatches per second\n        max_burst: the new maximum burst size\n        max_concurrent: the new maximum number of concurrent dispatches\n        max_attempts: the new maximum number of retries attempted\n    Returns:\n        The updated queue.\n    '
    client = tasks.CloudTasksClient()
    update_mask = {'paths': []}
    if uri:
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
        queue.http_target = http_target
        update_mask['paths'].append('http_target')
    if max_per_second != 0.0:
        queue.rate_limits.max_dispatches_per_second = max_per_second
    if max_burst != 0:
        queue.rate_limits.max_burst_size = max_burst
    if max_concurrent != 0:
        queue.rate_limits.max_concurrent_dispatches = max_concurrent
    update_mask['paths'].append('rate_limits')
    if max_attempts != 0:
        queue.retry_config.max_attempts = max_attempts
    update_mask['paths'].append('retry_config')
    request = tasks.UpdateQueueRequest(queue=queue, update_mask=update_mask)
    updated_queue = client.update_queue(request)
    return updated_queue