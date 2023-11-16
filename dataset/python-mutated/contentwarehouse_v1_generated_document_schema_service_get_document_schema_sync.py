from google.cloud import contentwarehouse_v1

def sample_get_document_schema():
    if False:
        print('Hello World!')
    client = contentwarehouse_v1.DocumentSchemaServiceClient()
    request = contentwarehouse_v1.GetDocumentSchemaRequest(name='name_value')
    response = client.get_document_schema(request=request)
    print(response)