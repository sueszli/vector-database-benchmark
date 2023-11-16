from google.cloud import contentwarehouse_v1

def sample_delete_document():
    if False:
        for i in range(10):
            print('nop')
    client = contentwarehouse_v1.DocumentServiceClient()
    request = contentwarehouse_v1.DeleteDocumentRequest(name='name_value')
    client.delete_document(request=request)