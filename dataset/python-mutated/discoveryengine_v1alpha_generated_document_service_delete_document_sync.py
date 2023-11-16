from google.cloud import discoveryengine_v1alpha

def sample_delete_document():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1alpha.DocumentServiceClient()
    request = discoveryengine_v1alpha.DeleteDocumentRequest(name='name_value')
    client.delete_document(request=request)