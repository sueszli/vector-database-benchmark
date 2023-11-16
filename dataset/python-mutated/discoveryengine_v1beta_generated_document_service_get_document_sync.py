from google.cloud import discoveryengine_v1beta

def sample_get_document():
    if False:
        print('Hello World!')
    client = discoveryengine_v1beta.DocumentServiceClient()
    request = discoveryengine_v1beta.GetDocumentRequest(name='name_value')
    response = client.get_document(request=request)
    print(response)