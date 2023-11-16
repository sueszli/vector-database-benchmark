from google.cloud import essential_contacts_v1

def sample_list_contacts():
    if False:
        i = 10
        return i + 15
    client = essential_contacts_v1.EssentialContactsServiceClient()
    request = essential_contacts_v1.ListContactsRequest(parent='parent_value')
    page_result = client.list_contacts(request=request)
    for response in page_result:
        print(response)