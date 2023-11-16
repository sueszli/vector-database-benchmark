from google.cloud import contentwarehouse

def update_document_schema(project_number: str, location: str, document_schema_id: str) -> None:
    if False:
        while True:
            i = 10
    document_schema_client = contentwarehouse.DocumentSchemaServiceClient()
    document_schema_path = document_schema_client.document_schema_path(project=project_number, location=location, document_schema=document_schema_id)
    updated_property_definition = contentwarehouse.PropertyDefinition(name='stock_symbol', display_name='Searchable text', is_searchable=True, is_repeatable=False, is_required=True, text_type_options=contentwarehouse.TextTypeOptions())
    update_document_schema_request = contentwarehouse.UpdateDocumentSchemaRequest(name=document_schema_path, document_schema=contentwarehouse.DocumentSchema(display_name='My Test Schema', property_definitions=[updated_property_definition]))
    updated_document_schema = document_schema_client.update_document_schema(request=update_document_schema_request)
    print(f'Updated Document Schema: {updated_document_schema}')