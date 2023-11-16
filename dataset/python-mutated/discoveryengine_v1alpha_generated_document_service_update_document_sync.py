from google.cloud import discoveryengine_v1alpha

def sample_update_document():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1alpha.DocumentServiceClient()
    request = discoveryengine_v1alpha.UpdateDocumentRequest()
    response = client.update_document(request=request)
    print(response)