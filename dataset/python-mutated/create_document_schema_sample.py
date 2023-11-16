from google.cloud import contentwarehouse

def sample_create_document_schema(project_number: str, location: str) -> None:
    if False:
        i = 10
        return i + 15
    'Creates document schema.\n\n    Args:\n        project_number: Google Cloud project number.\n        location: Google Cloud project location.\n    Returns:\n        Response object.\n    '
    document_schema_client = contentwarehouse.DocumentSchemaServiceClient()
    property_definition = contentwarehouse.PropertyDefinition(name='stock_symbol', display_name='Searchable text', is_searchable=True, text_type_options=contentwarehouse.TextTypeOptions())
    document_schema = contentwarehouse.DocumentSchema(display_name='My Test Schema', property_definitions=[property_definition])
    request = contentwarehouse.CreateDocumentSchemaRequest(parent=document_schema_client.common_location_path(project_number, location), document_schema=document_schema)
    response = document_schema_client.create_document_schema(request=request)
    print('Document Schema Created:', response)
    return response