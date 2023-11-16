from google.cloud import domains_v1beta1

def sample_list_registrations():
    if False:
        i = 10
        return i + 15
    client = domains_v1beta1.DomainsClient()
    request = domains_v1beta1.ListRegistrationsRequest(parent='parent_value')
    page_result = client.list_registrations(request=request)
    for response in page_result:
        print(response)