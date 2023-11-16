from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.NetworkEdgeSecurityServicesClient()
    request = compute_v1.DeleteNetworkEdgeSecurityServiceRequest(network_edge_security_service='network_edge_security_service_value', project='project_value', region='region_value')
    response = client.delete(request=request)
    print(response)