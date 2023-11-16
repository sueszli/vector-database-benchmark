from typing import Optional
from google.cloud import contentwarehouse

def sample_create_document(project_number: str, location: str, raw_document_path: str, raw_document_file_type: contentwarehouse.RawDocumentFileType, document_schema_id: str, user_id: str, reference_id: Optional[str]=None) -> contentwarehouse.CreateDocumentResponse:
    if False:
        for i in range(10):
            print('nop')
    'Creates a document.\n\n    Args:\n        project_number: Google Cloud project number.\n        location: Google Cloud project location.\n        raw_document_path: Raw document file in Cloud Storage path.\n        raw_document_file_type: Document file type\n                                https://cloud.google.com/python/docs/\n                                reference/contentwarehouse/latest/\n                                google.cloud.contentwarehouse_v1.types.RawDocumentFileType.\n        document_schema_id: Unique identifier for document schema.\n        user_id: user:YOUR_SERVICE_ACCOUNT_ID or user:USER_EMAIL.\n        reference_id: Identifier, must be unique per project and location.\n    Returns:\n        Response object.\n    '
    client = contentwarehouse.DocumentServiceClient()
    document_schema_name = client.document_schema_path(project=project_number, location=location, document_schema=document_schema_id)
    document = contentwarehouse.Document(raw_document_path=raw_document_path, display_name='Order Invoice', plain_text='Sample Invoice Document', raw_document_file_type=raw_document_file_type, document_schema_name=document_schema_name, reference_id=reference_id)
    request_metadata = contentwarehouse.RequestMetadata(user_info=contentwarehouse.UserInfo(id=user_id))
    parent = client.common_location_path(project=project_number, location=location)
    request = contentwarehouse.CreateDocumentRequest(parent=parent, request_metadata=request_metadata, document=document)
    response = client.create_document(request=request)
    return response