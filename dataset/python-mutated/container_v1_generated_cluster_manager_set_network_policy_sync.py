from google.cloud import container_v1

def sample_set_network_policy():
    if False:
        return 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetNetworkPolicyRequest()
    response = client.set_network_policy(request=request)
    print(response)