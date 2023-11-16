from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.SubnetworksClient()
    request = compute_v1.DeleteSubnetworkRequest(project='project_value', region='region_value', subnetwork='subnetwork_value')
    response = client.delete(request=request)
    print(response)