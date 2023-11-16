from google.cloud import contentwarehouse_v1

def sample_delete_document_link():
    if False:
        while True:
            i = 10
    client = contentwarehouse_v1.DocumentLinkServiceClient()
    request = contentwarehouse_v1.DeleteDocumentLinkRequest(name='name_value')
    client.delete_document_link(request=request)