from google.cloud import dataproc_v1

def sample_delete_autoscaling_policy():
    if False:
        return 10
    client = dataproc_v1.AutoscalingPolicyServiceClient()
    request = dataproc_v1.DeleteAutoscalingPolicyRequest(name='name_value')
    client.delete_autoscaling_policy(request=request)