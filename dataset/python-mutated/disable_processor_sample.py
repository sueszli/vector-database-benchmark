from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import FailedPrecondition
from google.cloud import documentai

def disable_processor_sample(project_id: str, location: str, processor_id: str) -> None:
    if False:
        while True:
            i = 10
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    processor_name = client.processor_path(project_id, location, processor_id)
    request = documentai.DisableProcessorRequest(name=processor_name)
    try:
        operation = client.disable_processor(request=request)
        print(operation.operation.name)
        operation.result()
    except FailedPrecondition as e:
        print(e.message)