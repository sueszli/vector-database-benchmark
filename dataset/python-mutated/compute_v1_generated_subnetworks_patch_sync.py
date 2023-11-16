from google.cloud import compute_v1

def sample_patch():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.SubnetworksClient()
    request = compute_v1.PatchSubnetworkRequest(project='project_value', region='region_value', subnetwork='subnetwork_value')
    response = client.patch(request=request)
    print(response)