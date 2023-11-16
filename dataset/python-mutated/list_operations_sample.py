from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.longrunning.operations_pb2 import ListOperationsRequest

def list_operations_sample(project_id: str, location: str, operations_filter: str) -> None:
    if False:
        return 10
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    name = client.common_location_path(project=project_id, location=location)
    request = ListOperationsRequest(name=f'{name}/operations', filter=operations_filter)
    operations = client.list_operations(request=request)
    print(operations)