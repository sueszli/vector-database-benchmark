from typing import Optional
from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def process_document_sample(project_id: str, location: str, processor_id: str, file_path: str, mime_type: str, field_mask: Optional[str]=None, processor_version_id: Optional[str]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    if processor_version_id:
        name = client.processor_version_path(project_id, location, processor_id, processor_version_id)
    else:
        name = client.processor_path(project_id, location, processor_id)
    with open(file_path, 'rb') as image:
        image_content = image.read()
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
    process_options = documentai.ProcessOptions(individual_page_selector=documentai.ProcessOptions.IndividualPageSelector(pages=[1]))
    request = documentai.ProcessRequest(name=name, raw_document=raw_document, field_mask=field_mask, process_options=process_options)
    result = client.process_document(request=request)
    document = result.document
    print('The document contains the following text:')
    print(document.text)