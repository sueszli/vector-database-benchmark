from google.cloud import contentwarehouse

def sample_get_document_schema(project_number: str, location: str, document_schema_id: str) -> None:
    if False:
        while True:
            i = 10
    'Gets document schema details.\n\n    Args:\n        project_number: Google Cloud project number.\n        location: Google Cloud project location.\n        document_schema_id: Unique identifier for document schema\n    Returns:\n        Response object.\n    '
    document_schema_client = contentwarehouse.DocumentSchemaServiceClient()
    document_schema_path = document_schema_client.document_schema_path(project=project_number, location=location, document_schema=document_schema_id)
    request = contentwarehouse.GetDocumentSchemaRequest(name=document_schema_path)
    response = document_schema_client.get_document_schema(request=request)
    print('Document Schema:', response)
    return response