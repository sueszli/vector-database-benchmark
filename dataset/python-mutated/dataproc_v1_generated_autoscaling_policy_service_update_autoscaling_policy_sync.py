from google.cloud import dataproc_v1

def sample_update_autoscaling_policy():
    if False:
        i = 10
        return i + 15
    client = dataproc_v1.AutoscalingPolicyServiceClient()
    policy = dataproc_v1.AutoscalingPolicy()
    policy.basic_algorithm.yarn_config.scale_up_factor = 0.1578
    policy.basic_algorithm.yarn_config.scale_down_factor = 0.1789
    policy.worker_config.max_instances = 1389
    request = dataproc_v1.UpdateAutoscalingPolicyRequest(policy=policy)
    response = client.update_autoscaling_policy(request=request)
    print(response)