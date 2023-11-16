from typing import List
from google.cloud import logging_v2

def list_logs(project_id: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Lists all logs in a project.\n\n    Args:\n        project_id: the ID of the project\n\n    Returns:\n        A list of log names.\n    '
    client = logging_v2.services.logging_service_v2.LoggingServiceV2Client()
    request = logging_v2.types.ListLogsRequest(parent=f'projects/{project_id}')
    logs = client.list_logs(request=request)
    for log in logs:
        print(log)
    return logs