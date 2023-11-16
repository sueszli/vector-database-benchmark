from google.cloud import service_usage_v1

def sample_disable_service():
    if False:
        return 10
    client = service_usage_v1.ServiceUsageClient()
    request = service_usage_v1.DisableServiceRequest()
    operation = client.disable_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)