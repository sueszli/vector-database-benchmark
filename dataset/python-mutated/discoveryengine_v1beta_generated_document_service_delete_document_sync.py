from google.cloud import discoveryengine_v1beta

def sample_delete_document():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1beta.DocumentServiceClient()
    request = discoveryengine_v1beta.DeleteDocumentRequest(name='name_value')
    client.delete_document(request=request)