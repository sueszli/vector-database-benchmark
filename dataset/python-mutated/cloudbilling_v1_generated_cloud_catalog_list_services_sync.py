from google.cloud import billing_v1

def sample_list_services():
    if False:
        print('Hello World!')
    client = billing_v1.CloudCatalogClient()
    request = billing_v1.ListServicesRequest()
    page_result = client.list_services(request=request)
    for response in page_result:
        print(response)