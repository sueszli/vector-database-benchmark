from google.cloud import gke_multicloud_v1

def sample_get_attached_server_config():
    if False:
        while True:
            i = 10
    client = gke_multicloud_v1.AttachedClustersClient()
    request = gke_multicloud_v1.GetAttachedServerConfigRequest(name='name_value')
    response = client.get_attached_server_config(request=request)
    print(response)