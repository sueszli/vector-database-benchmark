from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.InstanceTemplatesClient()
    request = compute_v1.DeleteInstanceTemplateRequest(instance_template='instance_template_value', project='project_value')
    response = client.delete(request=request)
    print(response)