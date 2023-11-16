from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.HealthChecksClient()
    request = compute_v1.DeleteHealthCheckRequest(health_check='health_check_value', project='project_value')
    response = client.delete(request=request)
    print(response)