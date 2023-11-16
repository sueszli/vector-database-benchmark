from google.cloud import discoveryengine_v1beta

def sample_list_documents():
    if False:
        print('Hello World!')
    client = discoveryengine_v1beta.DocumentServiceClient()
    request = discoveryengine_v1beta.ListDocumentsRequest(parent='parent_value')
    page_result = client.list_documents(request=request)
    for response in page_result:
        print(response)