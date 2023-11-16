from google.cloud import gke_multicloud_v1

def sample_create_attached_cluster():
    if False:
        i = 10
        return i + 15
    client = gke_multicloud_v1.AttachedClustersClient()
    attached_cluster = gke_multicloud_v1.AttachedCluster()
    attached_cluster.platform_version = 'platform_version_value'
    attached_cluster.distribution = 'distribution_value'
    attached_cluster.fleet.project = 'project_value'
    request = gke_multicloud_v1.CreateAttachedClusterRequest(parent='parent_value', attached_cluster=attached_cluster, attached_cluster_id='attached_cluster_id_value')
    operation = client.create_attached_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)