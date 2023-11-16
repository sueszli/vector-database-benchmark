from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.AutoscalersClient()
    request = compute_v1.DeleteAutoscalerRequest(autoscaler='autoscaler_value', project='project_value', zone='zone_value')
    response = client.delete(request=request)
    print(response)