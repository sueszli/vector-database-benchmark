from google.cloud import dialogflow_v2beta1

def sample_import_documents():
    if False:
        return 10
    client = dialogflow_v2beta1.DocumentsClient()
    gcs_source = dialogflow_v2beta1.GcsSources()
    gcs_source.uris = ['uris_value1', 'uris_value2']
    document_template = dialogflow_v2beta1.ImportDocumentTemplate()
    document_template.mime_type = 'mime_type_value'
    document_template.knowledge_types = ['SMART_REPLY']
    request = dialogflow_v2beta1.ImportDocumentsRequest(gcs_source=gcs_source, parent='parent_value', document_template=document_template)
    operation = client.import_documents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)