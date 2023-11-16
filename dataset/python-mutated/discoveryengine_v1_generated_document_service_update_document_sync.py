from google.cloud import discoveryengine_v1

def sample_update_document():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1.DocumentServiceClient()
    request = discoveryengine_v1.UpdateDocumentRequest()
    response = client.update_document(request=request)
    print(response)