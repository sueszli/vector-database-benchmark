from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.NetworkEdgeSecurityServicesClient()
    request = compute_v1.GetNetworkEdgeSecurityServiceRequest(network_edge_security_service='network_edge_security_service_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)