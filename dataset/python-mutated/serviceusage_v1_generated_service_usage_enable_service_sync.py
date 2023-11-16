from google.cloud import service_usage_v1

def sample_enable_service():
    if False:
        while True:
            i = 10
    client = service_usage_v1.ServiceUsageClient()
    request = service_usage_v1.EnableServiceRequest()
    operation = client.enable_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)