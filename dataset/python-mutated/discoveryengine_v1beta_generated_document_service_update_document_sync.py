from google.cloud import discoveryengine_v1beta

def sample_update_document():
    if False:
        return 10
    client = discoveryengine_v1beta.DocumentServiceClient()
    request = discoveryengine_v1beta.UpdateDocumentRequest()
    response = client.update_document(request=request)
    print(response)