from google.cloud import container_v1beta1

def sample_rollback_node_pool_upgrade():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.RollbackNodePoolUpgradeRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', node_pool_id='node_pool_id_value')
    response = client.rollback_node_pool_upgrade(request=request)
    print(response)