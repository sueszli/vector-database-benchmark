from google.cloud import gke_multicloud_v1

def sample_update_azure_cluster():
    if False:
        for i in range(10):
            print('nop')
    client = gke_multicloud_v1.AzureClustersClient()
    azure_cluster = gke_multicloud_v1.AzureCluster()
    azure_cluster.azure_region = 'azure_region_value'
    azure_cluster.resource_group_id = 'resource_group_id_value'
    azure_cluster.networking.virtual_network_id = 'virtual_network_id_value'
    azure_cluster.networking.pod_address_cidr_blocks = ['pod_address_cidr_blocks_value1', 'pod_address_cidr_blocks_value2']
    azure_cluster.networking.service_address_cidr_blocks = ['service_address_cidr_blocks_value1', 'service_address_cidr_blocks_value2']
    azure_cluster.control_plane.version = 'version_value'
    azure_cluster.control_plane.ssh_config.authorized_key = 'authorized_key_value'
    azure_cluster.authorization.admin_users.username = 'username_value'
    azure_cluster.fleet.project = 'project_value'
    request = gke_multicloud_v1.UpdateAzureClusterRequest(azure_cluster=azure_cluster)
    operation = client.update_azure_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)