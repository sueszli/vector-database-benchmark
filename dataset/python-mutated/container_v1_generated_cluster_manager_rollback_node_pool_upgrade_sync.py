from google.cloud import container_v1

def sample_rollback_node_pool_upgrade():
    if False:
        return 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.RollbackNodePoolUpgradeRequest()
    response = client.rollback_node_pool_upgrade(request=request)
    print(response)