from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import FailedPrecondition
from google.api_core.exceptions import InvalidArgument
from google.cloud import documentai

def undeploy_processor_version_sample(project_id: str, location: str, processor_id: str, processor_version_id: str) -> None:
    if False:
        print('Hello World!')
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    name = client.processor_version_path(project_id, location, processor_id, processor_version_id)
    try:
        operation = client.undeploy_processor_version(name=name)
        print(operation.operation.name)
        operation.result()
    except (FailedPrecondition, InvalidArgument) as e:
        print(e.message)