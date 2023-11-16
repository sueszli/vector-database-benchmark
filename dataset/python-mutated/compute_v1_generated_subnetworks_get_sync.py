from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.SubnetworksClient()
    request = compute_v1.GetSubnetworkRequest(project='project_value', region='region_value', subnetwork='subnetwork_value')
    response = client.get(request=request)
    print(response)