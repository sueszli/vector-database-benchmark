from google.cloud import contentwarehouse

def sample_delete_document(document_name: str, user_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Deletes a document.\n\n    Args:\n        document_name: The resource name of the document.\n                    Format: 'projects/{project_number}/\n                    locations/{location}/documents/{document_id}'.\n        user_id: user:YOUR_SERVICE_ACCOUNT_ID or user:USER_EMAIL.\n    Returns:\n        None, if operation is successful.\n    "
    client = contentwarehouse.DocumentServiceClient()
    request_metadata = contentwarehouse.RequestMetadata(user_info=contentwarehouse.UserInfo(id=user_id))
    request = contentwarehouse.DeleteDocumentRequest(name=document_name, request_metadata=request_metadata)
    response = client.delete_document(request=request)
    return response