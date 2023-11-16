from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import NotFound
from google.cloud import documentai

def set_default_processor_version_sample(project_id: str, location: str, processor_id: str, processor_version_id: str) -> None:
    if False:
        print('Hello World!')
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    processor = client.processor_path(project_id, location, processor_id)
    processor_version = client.processor_version_path(project_id, location, processor_id, processor_version_id)
    request = documentai.SetDefaultProcessorVersionRequest(processor=processor, default_processor_version=processor_version)
    try:
        operation = client.set_default_processor_version(request)
        print(operation.operation.name)
        operation.result()
    except NotFound as e:
        print(e.message)