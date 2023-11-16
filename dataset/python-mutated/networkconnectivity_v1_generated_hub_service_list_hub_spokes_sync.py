from google.cloud import networkconnectivity_v1

def sample_list_hub_spokes():
    if False:
        i = 10
        return i + 15
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.ListHubSpokesRequest(name='name_value')
    page_result = client.list_hub_spokes(request=request)
    for response in page_result:
        print(response)