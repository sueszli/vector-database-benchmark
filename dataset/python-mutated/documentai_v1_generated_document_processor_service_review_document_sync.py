from google.cloud import documentai_v1

def sample_review_document():
    if False:
        i = 10
        return i + 15
    client = documentai_v1.DocumentProcessorServiceClient()
    inline_document = documentai_v1.Document()
    inline_document.uri = 'uri_value'
    request = documentai_v1.ReviewDocumentRequest(inline_document=inline_document, human_review_config='human_review_config_value')
    operation = client.review_document(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)