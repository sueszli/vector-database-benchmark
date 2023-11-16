from google.cloud import vmwareengine_v1

def sample_update_private_cloud():
    if False:
        return 10
    client = vmwareengine_v1.VmwareEngineClient()
    private_cloud = vmwareengine_v1.PrivateCloud()
    private_cloud.network_config.management_cidr = 'management_cidr_value'
    private_cloud.management_cluster.cluster_id = 'cluster_id_value'
    request = vmwareengine_v1.UpdatePrivateCloudRequest(private_cloud=private_cloud)
    operation = client.update_private_cloud(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)