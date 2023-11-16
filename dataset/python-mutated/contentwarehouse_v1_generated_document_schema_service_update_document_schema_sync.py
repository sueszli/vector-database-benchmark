from google.cloud import contentwarehouse_v1

def sample_update_document_schema():
    if False:
        print('Hello World!')
    client = contentwarehouse_v1.DocumentSchemaServiceClient()
    document_schema = contentwarehouse_v1.DocumentSchema()
    document_schema.display_name = 'display_name_value'
    request = contentwarehouse_v1.UpdateDocumentSchemaRequest(name='name_value', document_schema=document_schema)
    response = client.update_document_schema(request=request)
    print(response)