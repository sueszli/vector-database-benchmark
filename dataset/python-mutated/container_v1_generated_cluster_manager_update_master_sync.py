from google.cloud import container_v1

def sample_update_master():
    if False:
        i = 10
        return i + 15
    client = container_v1.ClusterManagerClient()
    request = container_v1.UpdateMasterRequest(master_version='master_version_value')
    response = client.update_master(request=request)
    print(response)