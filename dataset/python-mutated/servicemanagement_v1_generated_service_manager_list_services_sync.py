from google.cloud import servicemanagement_v1

def sample_list_services():
    if False:
        for i in range(10):
            print('nop')
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.ListServicesRequest()
    page_result = client.list_services(request=request)
    for response in page_result:
        print(response)