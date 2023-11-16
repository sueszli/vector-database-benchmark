from google.cloud import gke_multicloud_v1

def sample_get_attached_cluster():
    if False:
        while True:
            i = 10
    client = gke_multicloud_v1.AttachedClustersClient()
    request = gke_multicloud_v1.GetAttachedClusterRequest(name='name_value')
    response = client.get_attached_cluster(request=request)
    print(response)