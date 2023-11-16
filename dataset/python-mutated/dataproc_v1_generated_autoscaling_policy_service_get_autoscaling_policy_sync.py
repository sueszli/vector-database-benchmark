from google.cloud import dataproc_v1

def sample_get_autoscaling_policy():
    if False:
        for i in range(10):
            print('nop')
    client = dataproc_v1.AutoscalingPolicyServiceClient()
    request = dataproc_v1.GetAutoscalingPolicyRequest(name='name_value')
    response = client.get_autoscaling_policy(request=request)
    print(response)