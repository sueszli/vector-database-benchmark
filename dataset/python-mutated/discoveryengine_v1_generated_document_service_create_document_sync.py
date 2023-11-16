from google.cloud import discoveryengine_v1

def sample_create_document():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1.DocumentServiceClient()
    request = discoveryengine_v1.CreateDocumentRequest(parent='parent_value', document_id='document_id_value')
    response = client.create_document(request=request)
    print(response)