from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def review_document_sample(project_id: str, location: str, processor_id: str, file_path: str, mime_type: str) -> None:
    if False:
        return 10
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    inline_document = process_document(project_id, location, processor_id, file_path, mime_type)
    human_review_config = client.human_review_config_path(project_id, location, processor_id)
    priority = documentai.ReviewDocumentRequest.Priority.DEFAULT
    request = documentai.ReviewDocumentRequest(inline_document=inline_document, human_review_config=human_review_config, enable_schema_validation=False, priority=priority)
    operation = client.review_document(request=request)
    print(operation.operation.name)

def process_document(project_id: str, location: str, processor_id: str, file_path: str, mime_type: str) -> documentai.Document:
    if False:
        while True:
            i = 10
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    name = client.processor_path(project_id, location, processor_id)
    with open(file_path, 'rb') as image:
        image_content = image.read()
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document