from google.cloud import contentwarehouse_v1

def sample_lock_document():
    if False:
        while True:
            i = 10
    client = contentwarehouse_v1.DocumentServiceClient()
    request = contentwarehouse_v1.LockDocumentRequest(name='name_value')
    response = client.lock_document(request=request)
    print(response)