from google.cloud import compute_v1

def sample_patch():
    if False:
        while True:
            i = 10
    client = compute_v1.AutoscalersClient()
    request = compute_v1.PatchAutoscalerRequest(project='project_value', zone='zone_value')
    response = client.patch(request=request)
    print(response)