from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def list_processors_sample(project_id: str, location: str) -> None:
    if False:
        print('Hello World!')
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    parent = client.common_location_path(project_id, location)
    processor_list = client.list_processors(parent=parent)
    for processor in processor_list:
        print(f'Processor Name: {processor.name}')
        print(f'Processor Display Name: {processor.display_name}')
        print(f'Processor Type: {processor.type_}')
        print('')