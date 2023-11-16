from google.cloud import compute_v1

def sample_update():
    if False:
        while True:
            i = 10
    client = compute_v1.HealthChecksClient()
    request = compute_v1.UpdateHealthCheckRequest(health_check='health_check_value', project='project_value')
    response = client.update(request=request)
    print(response)