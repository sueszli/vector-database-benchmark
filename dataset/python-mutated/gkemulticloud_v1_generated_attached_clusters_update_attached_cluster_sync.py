from google.cloud import gke_multicloud_v1

def sample_update_attached_cluster():
    if False:
        print('Hello World!')
    client = gke_multicloud_v1.AttachedClustersClient()
    attached_cluster = gke_multicloud_v1.AttachedCluster()
    attached_cluster.platform_version = 'platform_version_value'
    attached_cluster.distribution = 'distribution_value'
    attached_cluster.fleet.project = 'project_value'
    request = gke_multicloud_v1.UpdateAttachedClusterRequest(attached_cluster=attached_cluster)
    operation = client.update_attached_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)