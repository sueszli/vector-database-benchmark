from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def get_processor_version_sample(project_id: str, location: str, processor_id: str, processor_version_id: str) -> None:
    if False:
        i = 10
        return i + 15
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    name = client.processor_version_path(project_id, location, processor_id, processor_version_id)
    processor_version = client.get_processor_version(name=name)
    print(f'Processor Version: {processor_version_id}')
    print(f'Display Name: {processor_version.display_name}')
    print(processor_version.state)