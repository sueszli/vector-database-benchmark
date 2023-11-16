from google.cloud import container_v1

def sample_complete_node_pool_upgrade():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1.ClusterManagerClient()
    request = container_v1.CompleteNodePoolUpgradeRequest()
    client.complete_node_pool_upgrade(request=request)