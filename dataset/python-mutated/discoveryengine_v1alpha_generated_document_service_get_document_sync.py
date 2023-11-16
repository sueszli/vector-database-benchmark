from google.cloud import discoveryengine_v1alpha

def sample_get_document():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1alpha.DocumentServiceClient()
    request = discoveryengine_v1alpha.GetDocumentRequest(name='name_value')
    response = client.get_document(request=request)
    print(response)