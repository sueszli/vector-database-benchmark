from google.cloud import documentai_v1beta2

def sample_process_document():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta2.DocumentUnderstandingServiceClient()
    input_config = documentai_v1beta2.InputConfig()
    input_config.gcs_source.uri = 'uri_value'
    input_config.mime_type = 'mime_type_value'
    request = documentai_v1beta2.ProcessDocumentRequest(input_config=input_config)
    response = client.process_document(request=request)
    print(response)