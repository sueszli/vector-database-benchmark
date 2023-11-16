from google.cloud import gke_multicloud_v1

def sample_delete_attached_cluster():
    if False:
        print('Hello World!')
    client = gke_multicloud_v1.AttachedClustersClient()
    request = gke_multicloud_v1.DeleteAttachedClusterRequest(name='name_value')
    operation = client.delete_attached_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)