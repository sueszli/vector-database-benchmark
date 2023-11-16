from google.cloud import resourcemanager_v3

def sample_search_organizations():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.OrganizationsClient()
    request = resourcemanager_v3.SearchOrganizationsRequest()
    page_result = client.search_organizations(request=request)
    for response in page_result:
        print(response)