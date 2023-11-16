from google.cloud import documentai_v1beta3

def sample_review_document():
    if False:
        print('Hello World!')
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    inline_document = documentai_v1beta3.Document()
    inline_document.uri = 'uri_value'
    request = documentai_v1beta3.ReviewDocumentRequest(inline_document=inline_document, human_review_config='human_review_config_value')
    operation = client.review_document(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)