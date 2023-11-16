from google.cloud import compute_v1

def sample_resume():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.ResumeInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.resume(request=request)
    print(response)