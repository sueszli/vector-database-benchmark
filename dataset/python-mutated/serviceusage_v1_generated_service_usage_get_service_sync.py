from google.cloud import service_usage_v1

def sample_get_service():
    if False:
        for i in range(10):
            print('nop')
    client = service_usage_v1.ServiceUsageClient()
    request = service_usage_v1.GetServiceRequest()
    response = client.get_service(request=request)
    print(response)