from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def fetch_processor_types_sample(project_id: str, location: str) -> None:
    if False:
        while True:
            i = 10
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    parent = client.common_location_path(project_id, location)
    response = client.fetch_processor_types(parent=parent)
    print('Processor types:')
    for processor_type in response.processor_types:
        if processor_type.allow_creation:
            print(processor_type.type_)