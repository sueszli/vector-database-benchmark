from google.cloud import container_v1beta1

def sample_complete_node_pool_upgrade():
    if False:
        print('Hello World!')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.CompleteNodePoolUpgradeRequest()
    client.complete_node_pool_upgrade(request=request)