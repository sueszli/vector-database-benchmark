from google.cloud import compute_v1

def sample_set_private_ip_google_access():
    if False:
        i = 10
        return i + 15
    client = compute_v1.SubnetworksClient()
    request = compute_v1.SetPrivateIpGoogleAccessSubnetworkRequest(project='project_value', region='region_value', subnetwork='subnetwork_value')
    response = client.set_private_ip_google_access(request=request)
    print(response)