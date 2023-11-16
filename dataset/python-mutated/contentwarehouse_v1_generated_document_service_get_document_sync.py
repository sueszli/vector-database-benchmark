from google.cloud import contentwarehouse_v1

def sample_get_document():
    if False:
        for i in range(10):
            print('nop')
    client = contentwarehouse_v1.DocumentServiceClient()
    request = contentwarehouse_v1.GetDocumentRequest(name='name_value')
    response = client.get_document(request=request)
    print(response)