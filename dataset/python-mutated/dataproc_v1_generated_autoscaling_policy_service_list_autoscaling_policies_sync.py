from google.cloud import dataproc_v1

def sample_list_autoscaling_policies():
    if False:
        i = 10
        return i + 15
    client = dataproc_v1.AutoscalingPolicyServiceClient()
    request = dataproc_v1.ListAutoscalingPoliciesRequest(parent='parent_value')
    page_result = client.list_autoscaling_policies(request=request)
    for response in page_result:
        print(response)