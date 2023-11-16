from google.cloud import dialogflow_v2beta1

def sample_create_document():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.DocumentsClient()
    document = dialogflow_v2beta1.Document()
    document.content_uri = 'content_uri_value'
    document.display_name = 'display_name_value'
    document.mime_type = 'mime_type_value'
    document.knowledge_types = ['SMART_REPLY']
    request = dialogflow_v2beta1.CreateDocumentRequest(parent='parent_value', document=document)
    operation = client.create_document(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)