from google.cloud import compute_v1

def sample_list_peering_routes():
    if False:
        print('Hello World!')
    client = compute_v1.NetworksClient()
    request = compute_v1.ListPeeringRoutesNetworksRequest(network='network_value', project='project_value')
    page_result = client.list_peering_routes(request=request)
    for response in page_result:
        print(response)