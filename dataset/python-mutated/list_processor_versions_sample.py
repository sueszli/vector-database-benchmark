from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def list_processor_versions_sample(project_id: str, location: str, processor_id: str) -> None:
    if False:
        return 10
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    parent = client.processor_path(project_id, location, processor_id)
    processor_versions = client.list_processor_versions(parent=parent)
    for processor_version in processor_versions:
        processor_version_id = client.parse_processor_version_path(processor_version.name)['processor_version']
        print(f'Processor Version: {processor_version_id}')
        print(f'Display Name: {processor_version.display_name}')
        print(processor_version.state)
        print('')