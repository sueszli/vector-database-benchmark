from google.cloud import contentwarehouse_v1

def sample_create_document_link():
    if False:
        for i in range(10):
            print('nop')
    client = contentwarehouse_v1.DocumentLinkServiceClient()
    request = contentwarehouse_v1.CreateDocumentLinkRequest(parent='parent_value')
    response = client.create_document_link(request=request)
    print(response)