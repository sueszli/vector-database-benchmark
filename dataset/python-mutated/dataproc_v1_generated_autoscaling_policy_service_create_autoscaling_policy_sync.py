from google.cloud import dataproc_v1

def sample_create_autoscaling_policy():
    if False:
        while True:
            i = 10
    client = dataproc_v1.AutoscalingPolicyServiceClient()
    policy = dataproc_v1.AutoscalingPolicy()
    policy.basic_algorithm.yarn_config.scale_up_factor = 0.1578
    policy.basic_algorithm.yarn_config.scale_down_factor = 0.1789
    policy.worker_config.max_instances = 1389
    request = dataproc_v1.CreateAutoscalingPolicyRequest(parent='parent_value', policy=policy)
    response = client.create_autoscaling_policy(request=request)
    print(response)