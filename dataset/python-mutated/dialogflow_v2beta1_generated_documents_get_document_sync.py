from google.cloud import dialogflow_v2beta1

def sample_get_document():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.DocumentsClient()
    request = dialogflow_v2beta1.GetDocumentRequest(name='name_value')
    response = client.get_document(request=request)
    print(response)