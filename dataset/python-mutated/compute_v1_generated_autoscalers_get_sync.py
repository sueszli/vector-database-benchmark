from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.AutoscalersClient()
    request = compute_v1.GetAutoscalerRequest(autoscaler='autoscaler_value', project='project_value', zone='zone_value')
    response = client.get(request=request)
    print(response)