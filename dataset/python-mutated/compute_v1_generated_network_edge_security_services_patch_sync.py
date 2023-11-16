from google.cloud import compute_v1

def sample_patch():
    if False:
        print('Hello World!')
    client = compute_v1.NetworkEdgeSecurityServicesClient()
    request = compute_v1.PatchNetworkEdgeSecurityServiceRequest(network_edge_security_service='network_edge_security_service_value', project='project_value', region='region_value')
    response = client.patch(request=request)
    print(response)