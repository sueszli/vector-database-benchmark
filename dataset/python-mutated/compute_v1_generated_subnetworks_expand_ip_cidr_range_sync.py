from google.cloud import compute_v1

def sample_expand_ip_cidr_range():
    if False:
        while True:
            i = 10
    client = compute_v1.SubnetworksClient()
    request = compute_v1.ExpandIpCidrRangeSubnetworkRequest(project='project_value', region='region_value', subnetwork='subnetwork_value')
    response = client.expand_ip_cidr_range(request=request)
    print(response)