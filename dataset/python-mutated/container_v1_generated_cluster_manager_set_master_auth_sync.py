from google.cloud import container_v1

def sample_set_master_auth():
    if False:
        return 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetMasterAuthRequest(action='SET_USERNAME')
    response = client.set_master_auth(request=request)
    print(response)