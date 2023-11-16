from google.cloud import contentwarehouse

def create_folder(project_number: str, location: str, user_id: str) -> contentwarehouse.Document:
    if False:
        i = 10
        return i + 15
    document_schema_client = contentwarehouse.DocumentSchemaServiceClient()
    parent = document_schema_client.common_location_path(project=project_number, location=location)
    create_folder_schema_request = contentwarehouse.CreateDocumentSchemaRequest(parent=parent, document_schema=contentwarehouse.DocumentSchema(display_name='Test Folder Schema ', document_is_folder=True))
    folder_schema = document_schema_client.create_document_schema(request=create_folder_schema_request)
    folder_client = contentwarehouse.DocumentServiceClient()
    folder = contentwarehouse.Document(display_name='My Test Folder', document_schema_name=folder_schema.name)
    create_folder_request = contentwarehouse.CreateDocumentRequest(parent=parent, document=folder, request_metadata=contentwarehouse.RequestMetadata(user_info=contentwarehouse.UserInfo(id=user_id)))
    folder_response = folder_client.create_document(request=create_folder_request)
    print(f'Rule Engine Output: {folder_response.rule_engine_output}')
    print(f'Folder Created: {folder_response.document}')
    return folder_response

def create_document(project_number: str, location: str, user_id: str) -> contentwarehouse.Document:
    if False:
        for i in range(10):
            print('nop')
    document_schema_client = contentwarehouse.DocumentSchemaServiceClient()
    parent = document_schema_client.common_location_path(project=project_number, location=location)
    property_definition = contentwarehouse.PropertyDefinition(name='stock_symbol', display_name='Searchable text', is_searchable=True, text_type_options=contentwarehouse.TextTypeOptions())
    create_document_schema_request = contentwarehouse.CreateDocumentSchemaRequest(parent=parent, document_schema=contentwarehouse.DocumentSchema(display_name='My Test Schema', property_definitions=[property_definition]))
    document_schema = document_schema_client.create_document_schema(request=create_document_schema_request)
    document_client = contentwarehouse.DocumentServiceClient()
    parent = document_client.common_location_path(project=project_number, location=location)
    document_property = contentwarehouse.Property(name=document_schema.property_definitions[0].name, text_values=contentwarehouse.TextArray(values=['GOOG']))
    document = contentwarehouse.Document(display_name='My Test Document', document_schema_name=document_schema.name, plain_text="This is a sample of a document's text.", properties=[document_property])
    create_document_request = contentwarehouse.CreateDocumentRequest(parent=parent, document=document, request_metadata=contentwarehouse.RequestMetadata(user_info=contentwarehouse.UserInfo(id=user_id)))
    document_response = document_client.create_document(request=create_document_request)
    print(f'Rule Engine Output: {document_response.rule_engine_output}')
    print(f'Document Created: {document_response.document}')
    return document_response

def create_folder_link_document(project_number: str, location: str, user_id: str) -> None:
    if False:
        i = 10
        return i + 15
    folder = create_folder(project_number, location, user_id)
    document = create_document(project_number, location, user_id)
    link_client = contentwarehouse.DocumentLinkServiceClient()
    link = contentwarehouse.DocumentLink(source_document_reference=contentwarehouse.DocumentReference(document_name=folder.document.name), target_document_reference=contentwarehouse.DocumentReference(document_name=document.document.name))
    create_document_link_request = contentwarehouse.CreateDocumentLinkRequest(parent=folder.document.name, document_link=link, request_metadata=contentwarehouse.RequestMetadata(user_info=contentwarehouse.UserInfo(id=user_id)))
    create_link_response = link_client.create_document_link(request=create_document_link_request)
    print(f'Link Created: {create_link_response}')
    linked_targets_request = contentwarehouse.ListLinkedTargetsRequest(parent=folder.document.name, request_metadata=contentwarehouse.RequestMetadata(user_info=contentwarehouse.UserInfo(id=user_id)))
    linked_targets_response = link_client.list_linked_targets(request=linked_targets_request)
    print(f'Validate Link Created: {linked_targets_response}')