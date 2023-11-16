from google.cloud import servicedirectory_v1

def sample_list_endpoints():
    if False:
        print('Hello World!')
    client = servicedirectory_v1.RegistrationServiceClient()
    request = servicedirectory_v1.ListEndpointsRequest(parent='parent_value')
    page_result = client.list_endpoints(request=request)
    for response in page_result:
        print(response)