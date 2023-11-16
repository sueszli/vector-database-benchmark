from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def get_processor_sample(project_id: str, location: str, processor_id: str) -> None:
    if False:
        while True:
            i = 10
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    name = client.processor_path(project_id, location, processor_id)
    processor = client.get_processor(name=name)
    print(f'Processor Name: {processor.name}')
    print(f'Processor Display Name: {processor.display_name}')
    print(f'Processor Type: {processor.type_}')