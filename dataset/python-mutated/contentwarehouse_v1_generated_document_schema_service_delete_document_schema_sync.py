from google.cloud import contentwarehouse_v1

def sample_delete_document_schema():
    if False:
        print('Hello World!')
    client = contentwarehouse_v1.DocumentSchemaServiceClient()
    request = contentwarehouse_v1.DeleteDocumentSchemaRequest(name='name_value')
    client.delete_document_schema(request=request)