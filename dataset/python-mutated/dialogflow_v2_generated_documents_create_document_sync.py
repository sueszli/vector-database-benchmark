from google.cloud import dialogflow_v2

def sample_create_document():
    if False:
        return 10
    client = dialogflow_v2.DocumentsClient()
    document = dialogflow_v2.Document()
    document.content_uri = 'content_uri_value'
    document.display_name = 'display_name_value'
    document.mime_type = 'mime_type_value'
    document.knowledge_types = ['AGENT_FACING_SMART_REPLY']
    request = dialogflow_v2.CreateDocumentRequest(parent='parent_value', document=document)
    operation = client.create_document(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)