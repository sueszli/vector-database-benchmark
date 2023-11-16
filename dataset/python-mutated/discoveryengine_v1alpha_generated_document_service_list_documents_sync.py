from google.cloud import discoveryengine_v1alpha

def sample_list_documents():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1alpha.DocumentServiceClient()
    request = discoveryengine_v1alpha.ListDocumentsRequest(parent='parent_value')
    page_result = client.list_documents(request=request)
    for response in page_result:
        print(response)