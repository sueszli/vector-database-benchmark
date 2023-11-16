from google.cloud import contentwarehouse

def sample_list_document_schemas(project_number: str, location: str) -> None:
    if False:
        while True:
            i = 10
    'Lists document schemas.\n\n    Args:\n        project_number: Google Cloud project number.\n        location: Google Cloud project location.\n    '
    document_schema_client = contentwarehouse.DocumentSchemaServiceClient()
    parent = document_schema_client.common_location_path(project=project_number, location=location)
    request = contentwarehouse.ListDocumentSchemasRequest(parent=parent)
    page_result = document_schema_client.list_document_schemas(request=request)
    responses = []
    print('Document Schemas:')
    for response in page_result:
        print(response)
        responses.append(response)
    return responses