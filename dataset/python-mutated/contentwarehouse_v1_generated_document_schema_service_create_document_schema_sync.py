from google.cloud import contentwarehouse_v1

def sample_create_document_schema():
    if False:
        i = 10
        return i + 15
    client = contentwarehouse_v1.DocumentSchemaServiceClient()
    document_schema = contentwarehouse_v1.DocumentSchema()
    document_schema.display_name = 'display_name_value'
    request = contentwarehouse_v1.CreateDocumentSchemaRequest(parent='parent_value', document_schema=document_schema)
    response = client.create_document_schema(request=request)
    print(response)