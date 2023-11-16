from google.cloud import compute_v1

def sample_patch():
    if False:
        i = 10
        return i + 15
    client = compute_v1.HealthChecksClient()
    request = compute_v1.PatchHealthCheckRequest(health_check='health_check_value', project='project_value')
    response = client.patch(request=request)
    print(response)