from google.cloud import documentai_v1beta3

def sample_process_document():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    inline_document = documentai_v1beta3.Document()
    inline_document.uri = 'uri_value'
    request = documentai_v1beta3.ProcessRequest(inline_document=inline_document, name='name_value')
    response = client.process_document(request=request)
    print(response)