from google.cloud import contentwarehouse_v1

def sample_list_document_schemas():
    if False:
        print('Hello World!')
    client = contentwarehouse_v1.DocumentSchemaServiceClient()
    request = contentwarehouse_v1.ListDocumentSchemasRequest(parent='parent_value')
    page_result = client.list_document_schemas(request=request)
    for response in page_result:
        print(response)