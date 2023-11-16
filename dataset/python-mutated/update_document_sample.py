from google.cloud import contentwarehouse

def sample_update_document(document_name: str, document: contentwarehouse.Document, user_id: str) -> contentwarehouse.CreateDocumentResponse:
    if False:
        return 10
    "Updates a document.\n\n    Args:\n        document_name: The resource name of the document.\n                    Format: 'projects/{project_number}/\n                    locations/{location}/documents/{document_id}'.\n        document: Document AI Warehouse Document object..\n        user_id: user_id: user:YOUR_SERVICE_ACCOUNT_ID or user:USER_EMAIL.\n    Returns:\n        Response object.\n    "
    client = contentwarehouse.DocumentServiceClient()
    document.display_name = 'Updated Order Invoice'
    request_metadata = contentwarehouse.RequestMetadata(user_info=contentwarehouse.UserInfo(id=user_id))
    request = contentwarehouse.UpdateDocumentRequest(name=document_name, document=document, request_metadata=request_metadata)
    response = client.update_document(request=request)
    return response