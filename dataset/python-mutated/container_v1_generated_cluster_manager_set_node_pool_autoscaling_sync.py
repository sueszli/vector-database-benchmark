from google.cloud import container_v1

def sample_set_node_pool_autoscaling():
    if False:
        while True:
            i = 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetNodePoolAutoscalingRequest()
    response = client.set_node_pool_autoscaling(request=request)
    print(response)