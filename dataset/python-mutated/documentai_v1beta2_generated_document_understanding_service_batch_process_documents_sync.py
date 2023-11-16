from google.cloud import documentai_v1beta2

def sample_batch_process_documents():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta2.DocumentUnderstandingServiceClient()
    requests = documentai_v1beta2.ProcessDocumentRequest()
    requests.input_config.gcs_source.uri = 'uri_value'
    requests.input_config.mime_type = 'mime_type_value'
    request = documentai_v1beta2.BatchProcessDocumentsRequest(requests=requests)
    operation = client.batch_process_documents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)