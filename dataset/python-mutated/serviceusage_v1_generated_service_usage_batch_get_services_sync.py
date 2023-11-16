from google.cloud import service_usage_v1

def sample_batch_get_services():
    if False:
        while True:
            i = 10
    client = service_usage_v1.ServiceUsageClient()
    request = service_usage_v1.BatchGetServicesRequest()
    response = client.batch_get_services(request=request)
    print(response)