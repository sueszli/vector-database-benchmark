from __future__ import annotations
from typing import Any
from google.cloud import contentwarehouse

def set_acl(project_number: str, location: str, policy: dict[str, list[dict[str, Any]]], user_id: str, document_id: str='') -> None:
    if False:
        return 10
    'Sets access control policies on project or document level.\n\n    Args:\n        project_number: Google Cloud project number.\n        location: Google Cloud project location.\n        policy: ACL policy to set, in JSON format.\n        user_id: user:YOUR_SERVICE_ACCOUNT_ID.\n        document_id: Record id in Document AI Warehouse.\n    '
    client = contentwarehouse.DocumentServiceClient()
    if document_id:
        request = contentwarehouse.SetAclRequest(resource=client.document_path(project_number, location, document_id), request_metadata=contentwarehouse.RequestMetadata(user_info=contentwarehouse.UserInfo(id=user_id)))
    else:
        request = contentwarehouse.SetAclRequest(resource=client.common_project_path(project_number), project_owner=True)
    request.policy = policy
    response = client.set_acl(request=request)
    print(response)