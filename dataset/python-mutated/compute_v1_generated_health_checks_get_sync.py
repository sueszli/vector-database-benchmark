from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.HealthChecksClient()
    request = compute_v1.GetHealthCheckRequest(health_check='health_check_value', project='project_value')
    response = client.get(request=request)
    print(response)