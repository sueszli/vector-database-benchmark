import json
import requests
from core.constants import FLAGSMITH_SIGNATURE_HEADER
from core.signing import sign_payload
from django.conf import settings
from environments.dynamodb.migrator import IdentityMigrator
from task_processor.decorators import register_task_handler
from task_processor.models import TaskPriority

def _should_forward(project_id: int) -> bool:
    if False:
        return 10
    migrator = IdentityMigrator(project_id)
    return bool(migrator.is_migration_done)

@register_task_handler(queue_size=2000, priority=TaskPriority.LOW)
def forward_identity_request(request_method: str, headers: dict, project_id: int, query_params: dict=None, request_data: dict=None):
    if False:
        i = 10
        return i + 15
    if not _should_forward(project_id):
        return
    url = settings.EDGE_API_URL + 'identities/'
    headers = _get_headers(request_method, headers, json.dumps(request_data) if request_data else '')
    if request_method == 'POST':
        requests.post(url, data=json.dumps(request_data), headers=headers, timeout=5)
        return
    requests.get(url, params=query_params, headers=headers, timeout=5)

@register_task_handler(queue_size=2000, priority=TaskPriority.LOW)
def forward_trait_request(request_method: str, headers: dict, project_id: int, payload: dict):
    if False:
        for i in range(10):
            print('nop')
    return forward_trait_request_sync(request_method, headers, project_id, payload)

def forward_trait_request_sync(request_method: str, headers: dict, project_id: int, payload: dict):
    if False:
        print('Hello World!')
    if not _should_forward(project_id):
        return
    url = settings.EDGE_API_URL + 'traits/'
    payload = json.dumps(payload)
    requests.post(url, data=payload, headers=_get_headers(request_method, headers, payload), timeout=5)

@register_task_handler(queue_size=1000, priority=TaskPriority.LOW)
def forward_trait_requests(request_method: str, headers: str, project_id: int, payload: dict):
    if False:
        return 10
    for trait_data in payload:
        forward_trait_request_sync(request_method, headers, project_id, trait_data)

def _get_headers(request_method: str, headers: dict, payload: str='') -> dict:
    if False:
        print('Hello World!')
    headers = {k: v for (k, v) in headers.items()}
    if request_method == 'GET':
        headers.pop('Content-Length', None)
    signature = sign_payload(payload, settings.EDGE_REQUEST_SIGNING_KEY)
    headers[FLAGSMITH_SIGNATURE_HEADER] = signature
    return headers