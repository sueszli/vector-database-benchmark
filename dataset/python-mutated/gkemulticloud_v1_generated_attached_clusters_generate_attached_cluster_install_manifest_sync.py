from google.cloud import gke_multicloud_v1

def sample_generate_attached_cluster_install_manifest():
    if False:
        print('Hello World!')
    client = gke_multicloud_v1.AttachedClustersClient()
    request = gke_multicloud_v1.GenerateAttachedClusterInstallManifestRequest(parent='parent_value', attached_cluster_id='attached_cluster_id_value', platform_version='platform_version_value')
    response = client.generate_attached_cluster_install_manifest(request=request)
    print(response)