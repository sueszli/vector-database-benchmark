from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def create_processor_sample(project_id: str, location: str, processor_display_name: str, processor_type: str) -> None:
    if False:
        return 10
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    parent = client.common_location_path(project_id, location)
    processor = client.create_processor(parent=parent, processor=documentai.Processor(display_name=processor_display_name, type_=processor_type))
    print(f'Processor Name: {processor.name}')
    print(f'Processor Display Name: {processor.display_name}')
    print(f'Processor Type: {processor.type_}')