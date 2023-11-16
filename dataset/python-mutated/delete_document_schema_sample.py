from google.cloud import contentwarehouse

def sample_delete_document_schema(project_number: str, location: str, document_schema_id: str) -> None:
    if False:
        return 10
    'Deletes document schema.\n\n    Args:\n        project_number: Google Cloud project number.\n        location: Google Cloud project location.\n        document_schema_id: Unique identifier for document schema\n    Returns:\n        None, if operation is successful\n    '
    document_schema_client = contentwarehouse.DocumentSchemaServiceClient()
    document_schema_path = document_schema_client.document_schema_path(project=project_number, location=location, document_schema=document_schema_id)
    request = contentwarehouse.DeleteDocumentSchemaRequest(name=document_schema_path)
    response = document_schema_client.delete_document_schema(request=request)
    return response