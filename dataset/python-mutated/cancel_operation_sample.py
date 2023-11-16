from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.longrunning.operations_pb2 import CancelOperationRequest

def cancel_operation_sample(location: str, operation_name: str) -> None:
    if False:
        i = 10
        return i + 15
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    request = CancelOperationRequest(name=operation_name)
    client.cancel_operation(request=request)
    print(f'Operation {operation_name} cancelled')