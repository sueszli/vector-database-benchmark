from google.cloud import service_usage_v1

def sample_batch_enable_services():
    if False:
        i = 10
        return i + 15
    client = service_usage_v1.ServiceUsageClient()
    request = service_usage_v1.BatchEnableServicesRequest()
    operation = client.batch_enable_services(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)