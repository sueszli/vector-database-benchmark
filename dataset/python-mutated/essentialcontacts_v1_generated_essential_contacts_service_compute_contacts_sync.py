from google.cloud import essential_contacts_v1

def sample_compute_contacts():
    if False:
        return 10
    client = essential_contacts_v1.EssentialContactsServiceClient()
    request = essential_contacts_v1.ComputeContactsRequest(parent='parent_value')
    page_result = client.compute_contacts(request=request)
    for response in page_result:
        print(response)