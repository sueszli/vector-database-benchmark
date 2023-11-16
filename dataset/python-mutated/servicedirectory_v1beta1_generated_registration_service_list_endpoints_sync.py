from google.cloud import servicedirectory_v1beta1

def sample_list_endpoints():
    if False:
        i = 10
        return i + 15
    client = servicedirectory_v1beta1.RegistrationServiceClient()
    request = servicedirectory_v1beta1.ListEndpointsRequest(parent='parent_value')
    page_result = client.list_endpoints(request=request)
    for response in page_result:
        print(response)