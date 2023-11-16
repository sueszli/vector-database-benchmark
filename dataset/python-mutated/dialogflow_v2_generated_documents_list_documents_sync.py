from google.cloud import dialogflow_v2

def sample_list_documents():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.DocumentsClient()
    request = dialogflow_v2.ListDocumentsRequest(parent='parent_value')
    page_result = client.list_documents(request=request)
    for response in page_result:
        print(response)