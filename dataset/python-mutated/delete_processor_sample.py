from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import NotFound
from google.cloud import documentai

def delete_processor_sample(project_id: str, location: str, processor_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    processor_name = client.processor_path(project_id, location, processor_id)
    try:
        operation = client.delete_processor(name=processor_name)
        print(operation.operation.name)
        operation.result()
    except NotFound as e:
        print(e.message)