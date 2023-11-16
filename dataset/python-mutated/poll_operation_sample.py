from time import sleep
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.longrunning.operations_pb2 import GetOperationRequest

def poll_operation_sample(location: str, operation_name: str) -> None:
    if False:
        return 10
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    request = GetOperationRequest(name=operation_name)
    while True:
        operation = client.get_operation(request=request)
        print(operation)
        if operation.done:
            break
        sleep(10)