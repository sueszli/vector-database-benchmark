from google.cloud import managedidentities_v1

def sample_list_domains():
    if False:
        print('Hello World!')
    client = managedidentities_v1.ManagedIdentitiesServiceClient()
    request = managedidentities_v1.ListDomainsRequest(parent='parent_value')
    page_result = client.list_domains(request=request)
    for response in page_result:
        print(response)