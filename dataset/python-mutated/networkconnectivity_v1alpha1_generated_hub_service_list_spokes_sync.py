from google.cloud import networkconnectivity_v1alpha1

def sample_list_spokes():
    if False:
        while True:
            i = 10
    client = networkconnectivity_v1alpha1.HubServiceClient()
    request = networkconnectivity_v1alpha1.ListSpokesRequest(parent='parent_value')
    page_result = client.list_spokes(request=request)
    for response in page_result:
        print(response)