from google.cloud import dialogflow_v2

def sample_update_document():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.DocumentsClient()
    document = dialogflow_v2.Document()
    document.content_uri = 'content_uri_value'
    document.display_name = 'display_name_value'
    document.mime_type = 'mime_type_value'
    document.knowledge_types = ['AGENT_FACING_SMART_REPLY']
    request = dialogflow_v2.UpdateDocumentRequest(document=document)
    operation = client.update_document(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)