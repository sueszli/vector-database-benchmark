from google.cloud import service_usage_v1

def sample_list_services():
    if False:
        while True:
            i = 10
    client = service_usage_v1.ServiceUsageClient()
    request = service_usage_v1.ListServicesRequest()
    page_result = client.list_services(request=request)
    for response in page_result:
        print(response)