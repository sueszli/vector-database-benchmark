from google.cloud import gke_multicloud_v1

def sample_import_attached_cluster():
    if False:
        print('Hello World!')
    client = gke_multicloud_v1.AttachedClustersClient()
    request = gke_multicloud_v1.ImportAttachedClusterRequest(parent='parent_value', fleet_membership='fleet_membership_value', platform_version='platform_version_value', distribution='distribution_value')
    operation = client.import_attached_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)