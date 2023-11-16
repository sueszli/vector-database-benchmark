from google.cloud import compute_v1

def sample_update():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.AutoscalersClient()
    request = compute_v1.UpdateAutoscalerRequest(project='project_value', zone='zone_value')
    response = client.update(request=request)
    print(response)