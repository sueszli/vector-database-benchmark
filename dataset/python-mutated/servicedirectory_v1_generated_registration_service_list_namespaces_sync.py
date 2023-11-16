from google.cloud import servicedirectory_v1

def sample_list_namespaces():
    if False:
        i = 10
        return i + 15
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.ListNamespacesRequest(parent='parent_value')
    page_result = client.list_namespaces(request=request)
    for response in page_result:
        print(response)