from google.cloud import container_v1beta1

def sample_set_node_pool_autoscaling():
    if False:
        print('Hello World!')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.SetNodePoolAutoscalingRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', node_pool_id='node_pool_id_value')
    response = client.set_node_pool_autoscaling(request=request)
    print(response)