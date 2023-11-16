from google.cloud import domains_v1

def sample_list_registrations():
    if False:
        print('Hello World!')
    client = domains_v1.DomainsClient()
    request = domains_v1.ListRegistrationsRequest(parent='parent_value')
    page_result = client.list_registrations(request=request)
    for response in page_result:
        print(response)