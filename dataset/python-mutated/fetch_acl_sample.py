from google.cloud import contentwarehouse

def fetch_acl(project_number: str, location: str, user_id: str, document_id: str='') -> None:
    if False:
        for i in range(10):
            print('nop')
    'Fetches access control policies on project or document level.\n\n    Args:\n        project_number: Google Cloud project number.\n        location: Google Cloud project location.\n        user_id: user:YOUR_SERVICE_ACCOUNT_ID.\n        document_id: Record id in Document AI Warehouse.\n    '
    client = contentwarehouse.DocumentServiceClient()
    if document_id:
        request = contentwarehouse.FetchAclRequest(resource=client.document_path(project_number, location, document_id), request_metadata=contentwarehouse.RequestMetadata(user_info=contentwarehouse.UserInfo(id=user_id)))
    else:
        request = contentwarehouse.FetchAclRequest(resource=client.common_project_path(project_number), project_owner=True)
    response = client.fetch_acl(request)
    print(response)