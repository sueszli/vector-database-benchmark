from typing import List
from google.cloud import logging

def get_sink(project_id: str, sink_name: str) -> logging.Sink:
    if False:
        return 10
    'Retrieves the metadata for a Cloud Logging Sink.\n\n    Args:\n        project_id: the ID of the project\n        sink_name: the name of the sink\n\n    Returns:\n        A Cloud Logging Sink.\n    '
    client = logging.Client(project=project_id)
    sink = client.sink(sink_name)
    sink.reload()
    print(f'Name: {sink.name}')
    print(f'Destination: {sink.destination}')
    print(f'Filter: {sink.filter_}')
    return sink